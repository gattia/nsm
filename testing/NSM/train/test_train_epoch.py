"""
``train_epoch``: the config it reads, the epoch it runs, and the metrics it returns.

Several contracts here share one shape: a bad config value must raise an error that names
its key before the value is used.
"""

import contextlib
import logging

import pytest
import torch

import NSM.train.train_deep_sdf as trainer
from NSM.train.train_deep_sdf import (
    _code_regularization_loss,
    _schedule_free_eval_warmup,
    train_epoch,
)
from NSM.utils import get_latent_vecs, get_learning_rate_schedules, get_optimizer

LATENT_SIZE = 4
N_SUBJECTS = 4
N_SAMPLES = 8


class LinearDecoder(torch.nn.Module):
    """One output column per surface from the concatenated ``[latent | xyz]`` input."""

    def __init__(self, surfaces=2):
        super().__init__()
        self.linear = torch.nn.Linear(LATENT_SIZE + 3, surfaces)

    def forward(self, x, epoch=None):
        return self.linear(x)


class TinyDataset(torch.utils.data.Dataset):
    """``(sample_dict, index)`` in the shape ``MultiSurfaceSDFSamples`` yields."""

    def __init__(self, surfaces=2, n_subjects=N_SUBJECTS):
        generator = torch.Generator().manual_seed(0)
        self.xyz = torch.rand(n_subjects, N_SAMPLES, 3, generator=generator)
        self.gt_sdf = torch.rand(n_subjects, N_SAMPLES, surfaces, generator=generator) - 0.5

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, index):
        return {"xyz": self.xyz[index], "gt_sdf": self.gt_sdf[index]}, index


def epoch_inputs(
    surfaces=2,
    objects_per_batch=2,
    n_subjects=N_SUBJECTS,
    model_lr=1e-3,
    latent_lr=1e-3,
    **overrides,
):
    """``(model, data_loader, latent_vecs, optimizer, config)`` for one seeded CPU epoch."""
    config = {
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "device": "cpu",
        "batch_split": 1,
        "samples_per_object_per_batch": N_SAMPLES,
        "enforce_minmax": False,
        "clamp_dist": 1.0,
        "surface_accuracy_e": None,
        "sample_difficulty_weight": None,
        "code_regularization": True,
        "code_regularization_type_prior": "identity",
        "code_regularization_weight": 1e-4,
        "code_regularization_warmup": 2,
        "code_cyclic_anneal": False,
        "n_epochs": 4,
        "grad_clip": None,
        "log_latent": None,
        "latent_size": LATENT_SIZE,
        "latent_bound": 10,
        "latent_init_std": 0.01,
        "latent_init_normal": True,
        "variational": False,
        "LearningRateSchedule": [
            {"Target": "model", "Type": "Constant", "Value": model_lr},
            {"Target": "latent", "Type": "Constant", "Value": latent_lr},
        ],
    }
    config.update(overrides)
    config["lr_schedules"] = get_learning_rate_schedules(config)
    dataset = TinyDataset(surfaces=surfaces, n_subjects=n_subjects)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=objects_per_batch, shuffle=False)
    torch.manual_seed(42)
    model = LinearDecoder(surfaces=surfaces)
    latent_vecs = get_latent_vecs(len(dataset), config)
    optimizer = get_optimizer(
        model, latent_vecs, lr_schedules=config["lr_schedules"], weight_decay=1e-4
    )
    return model, data_loader, latent_vecs, optimizer, config


def run_epoch(surfaces=2, **overrides):
    """One epoch's ``log_dict`` from a fresh seeded model and embedding."""
    model, data_loader, latent_vecs, optimizer, config = epoch_inputs(surfaces, **overrides)
    return train_epoch(
        model, data_loader, latent_vecs, optimizer, config, epoch=1, n_surfaces=surfaces
    )


class _StubScheduleFree:
    """The warm-up calls only ``eval()`` before the loop under test."""

    def eval(self):
        pass


def test_batch_split_changes_no_reported_number():
    """
    Fails if ``train_epoch`` reports a different ``loss``, ``mean_vec_length`` or
    ``std_vec_length`` at any ``batch_split`` than unsplit, or raises at a split count
    ``torch.chunk`` rounds down (KNOWN_ISSUES History 25).

    ``torch.chunk(t, k)`` returns at most ``k`` pieces, so 5, 7, 9 and 11 are the split
    counts that exercise it on this 16-row batch. At 6 and 16 a split holds a single row,
    whose ``torch.std`` is NaN. The ``surface_weighting`` case holds the weighted loss to
    the same invariance.
    """
    for surfaces, overrides in ((2, {}), (1, {}), (2, {"surface_weighting": [3, 1]})):
        reference = run_epoch(surfaces, **overrides)
        for splits in (2, 3, 4, 5, 6, 7, 8, 9, 11, 16):
            log = run_epoch(surfaces, batch_split=splits, **overrides)
            for key in ("loss", "mean_vec_length", "std_vec_length"):
                assert log[key] == pytest.approx(reference[key], rel=1e-6), (splits, key)


def test_the_warmup_feeds_the_decoder_what_the_epoch_feeds_it():
    """
    Fails if ``_schedule_free_eval_warmup`` feeds the decoder inputs that differ from what
    ``train_epoch`` feeds it at the same ``batch_split`` (#42).

    Both learning rates are 0, so the embedding cannot move and the inputs compare tensor
    for tensor. 5 and 7 are split counts ``torch.chunk`` rounds down on this 16-row batch.
    """
    for splits in (1, 2, 5, 7):
        recorded = {"epoch": [], "warmup": []}

        class Recording(LinearDecoder):
            def __init__(self, into):
                super().__init__()
                self.into = into

            def forward(self, x, epoch=None):
                self.into.append(x.detach().clone())
                return super().forward(x, epoch=epoch)

        frozen = dict(batch_split=splits, model_lr=0.0, latent_lr=0.0)
        _, loader, latents, optimizer, config = epoch_inputs(**frozen)
        train_epoch(
            Recording(recorded["epoch"]), loader, latents, optimizer, config, epoch=1, n_surfaces=2
        )
        _, loader, latents, _, config = epoch_inputs(**frozen)
        config["optimizer"] = "schedule_free_AdamW"
        _schedule_free_eval_warmup(
            Recording(recorded["warmup"]), latents, loader, _StubScheduleFree(), config, epoch=1
        )
        assert len(recorded["warmup"]) == len(recorded["epoch"]) > 0
        for from_epoch, from_warmup in zip(recorded["epoch"], recorded["warmup"]):
            assert torch.equal(from_epoch, from_warmup)


class TestTheLatentNormStatsAreTheEpochMean:
    def test_the_logged_mean_is_the_mean_over_batches_of_every_subject(self):
        """
        Fails if ``train_epoch`` logs the last batch's or the last split's mean latent norm
        as ``mean_vec_length`` instead of the mean over batches (#59; KNOWN_ISSUES History 12
        and 25).

        Latent LR 0, so the embedding cannot move and the answer is computable from it.
        Three subjects in batches of two, each split in two: ``[s0, s1], [s2]``. The last
        batch gives ``norm(s2) / 2`` and the last split ``(norm(s1) + norm(s2)) / 2``, so
        each wrong form differs from the mean.
        """
        model, loader, latents, optimizer, config = epoch_inputs(
            n_subjects=3, batch_split=2, latent_lr=0.0
        )
        norms = torch.norm(latents.weight.data, dim=1)
        expected = ((norms[0] + norms[1]) / 2 + norms[2]) / 2
        log = train_epoch(model, loader, latents, optimizer, config, epoch=1, n_surfaces=2)
        assert log["mean_vec_length"] == pytest.approx(expected.item(), rel=1e-6)


def test_config_values_are_refused_where_they_are_named():
    """
    Fails if ``train_epoch`` stops refusing ``multi_object_overlap``, a mis-sized
    ``surface_weighting`` or a ``samples_per_object_per_batch`` the batch disagrees with, or
    ``_code_regularization_loss`` stops refusing ``code_regularization_warmup=0``.

    ``multi_object_overlap`` must be refused before the first batch is fetched.
    ``_schedule_free_eval_warmup`` must refuse the sample count too.
    """

    class Unfetchable:
        def __iter__(self):
            raise AssertionError("the batch loop was entered")

        def __len__(self):
            return 1

    model, _, latents, optimizer, config = epoch_inputs(multi_object_overlap=True)
    with pytest.raises(NotImplementedError, match="multi_object_overlap"):
        train_epoch(model, Unfetchable(), latents, optimizer, config, epoch=1, n_surfaces=2)

    for weighting in ([1, 1, 1], [1]):
        with pytest.raises(ValueError, match="surface_weighting"):
            run_epoch(surface_weighting=weighting)

    for declared in (4, 16):
        with pytest.raises(ValueError, match="samples_per_object_per_batch"):
            run_epoch(samples_per_object_per_batch=declared)
        model, loader, latents, _, config = epoch_inputs(samples_per_object_per_batch=declared)
        config["optimizer"] = "schedule_free_AdamW"
        with pytest.raises(ValueError, match="samples_per_object_per_batch"):
            _schedule_free_eval_warmup(model, latents, loader, _StubScheduleFree(), config, 1)

    config = dict(config, code_regularization_warmup=0, code_regularization_type_prior="spherical")
    with pytest.raises(ValueError, match="code_regularization_weight"):
        _code_regularization_loss(
            batch_vecs=torch.randn(4, 8),
            mu=None,
            logvar=None,
            num_sdf_samples=100,
            epoch=0,
            config=config,
        )


class TestTheLogDict:
    """
    The configurations the regression baselines do not reach, held to identities rather
    than stored numbers.
    """

    def test_the_loss_is_its_parts_for_every_prior_and_surface_count(self):
        """
        Fails if ``train_epoch``'s ``loss`` is not ``l1_loss`` plus
        ``latent_code_regularization_loss``, or ``l1_loss`` is not the mean of the
        ``l1_loss_{i}`` under uniform weights, for any prior and one to three surfaces.

        It also fails if ``surface_weighting`` stops being normalized: ``[3, 1]`` must give
        the weighted mean, ``0.75 * l1_loss_0 + 0.25 * l1_loss_1``.
        """
        for prior in ("identity", "spherical", "kld_diagonal"):
            for surfaces in (1, 2, 3):
                log = run_epoch(surfaces, code_regularization_type_prior=prior)
                assert log["loss"] == pytest.approx(
                    log["l1_loss"] + log["latent_code_regularization_loss"], rel=1e-6
                )
                parts = [log[f"l1_loss_{index}"] for index in range(surfaces)]
                assert log["l1_loss"] == pytest.approx(sum(parts) / surfaces, rel=1e-6)

        weighted = run_epoch(surface_weighting=[3, 1])
        expected = 0.75 * weighted["l1_loss_0"] + 0.25 * weighted["l1_loss_1"]
        assert weighted["l1_loss"] == pytest.approx(expected, rel=1e-6)

    def test_the_documented_keys_are_there(self):
        """
        Fails if ``train_epoch``'s log dict for a two-surface run gains or loses a key,
        variational or not, such as a load-timing key from a dataset that timed no load.
        """
        for variational in (False, True):
            log = run_epoch(variational=variational, code_regularization_type_prior="kld_diagonal")
            assert set(log) == {
                "loss",
                "epoch_time_s",
                "l1_loss",
                "latent_code_regularization_loss",
                "mean_vec_length",
                "std_vec_length",
                "l1_loss_0",
                "l1_loss_1",
            }


@contextlib.contextmanager
def debug_records():
    """``NSM`` records at exactly DEBUG, without caplog, so absence is assertable."""
    collected = []

    class Collector(logging.Handler):
        def emit(self, record):
            if record.levelno == logging.DEBUG:
                collected.append(record.msg)

    handler, logger = Collector(logging.DEBUG), logging.getLogger("NSM")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield collected
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def test_a_host_at_debug_sees_the_records_whatever_the_config_says():
    """
    Fails if ``train_epoch`` emits a different set of DEBUG records when
    ``config["verbose"]`` is false, or stops emitting its ``"l1 loss: %s"`` record.

    The ``"l1 loss: %s"`` check keeps the set comparison from passing on two empty lists.
    """
    with debug_records() as default:
        run_epoch()
    with debug_records() as quiet:
        run_epoch(verbose=False)
    assert "l1 loss: %s" in default
    assert set(quiet) == set(default)


class TestGradClipReachesTheModelOnly:
    """
    ``grad_clip`` clips the decoder's parameters and never the latents, as upstream DeepSDF
    does. Ruled working by design (``SCOPE.md``).
    """

    def test_the_clipped_parameters_are_exactly_the_models(self, monkeypatch):
        """
        Fails if ``train_epoch`` with ``grad_clip`` hands ``clip_grad_norm_`` anything but
        exactly the decoder's parameters, such as the latent embedding.

        The spy replaces ``torch.nn.utils.clip_grad_norm_``. A trainer that imports the
        function by name bypasses it and fails here.
        """
        clipped = []
        real = torch.nn.utils.clip_grad_norm_

        def spy(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            clipped.append({id(p) for p in parameters})
            return real(parameters, max_norm, *args, **kwargs)

        monkeypatch.setattr(trainer.torch.nn.utils, "clip_grad_norm_", spy)
        model, loader, latents, optimizer, config = epoch_inputs(grad_clip=1.0)
        train_epoch(model, loader, latents, optimizer, config, epoch=1, n_surfaces=2)

        assert clipped and all(ids == {id(p) for p in model.parameters()} for ids in clipped)
        assert not {id(p) for p in latents.parameters()} & clipped[0]
