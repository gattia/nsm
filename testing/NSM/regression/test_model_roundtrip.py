"""
Checkpoint round-trip: train -> ``save_model`` -> ``load_model`` -> identical forward.

``testing/NSM/models/test_loader.py:232`` loads a saved model and never compares its
output to the original's, so a wrong-but-same-shaped forward passes every assertion it
makes. These tests compare the numbers.

The second half is the ``padding`` gap from ``SCOPE.md`` section 3.1: ``padding`` is not a
learned parameter, so a checkpoint trained at one value loads cleanly under strict
``load_state_dict`` at another and then samples the feature planes at the wrong scale --
silently, and with a measurable effect on the SDF.
"""

import inspect
import os

import numpy as np
import pytest
import torch
from _harness import ARCHITECTURE, LATENT_SIZE, build_model, training_config


def query_points(n=512, latent_size=LATENT_SIZE, seed=0):
    """Legacy concatenated input: ``[latent | xyz]``, the form ``train_epoch`` builds."""
    torch.manual_seed(seed)
    latent = torch.randn(1, latent_size).repeat(n, 1) * 0.05
    xyz = torch.rand(n, 3) * 2 - 1
    return torch.cat([latent, xyz], dim=1)


def forward(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(inputs)


def save_checkpoint(model, directory, epoch=1):
    from NSM.utils import save_model

    save_model({"experiment_directory": str(directory)}, epoch=epoch, decoder=model, optimizer=None)
    return os.path.join(str(directory), "model", f"{epoch}.pth")


class TestRoundTrip:
    def test_a_trained_model_round_trips_bitwise(self, reconstruction_model, tmp_path):
        """
        The requirement: what comes back out of a checkpoint must compute exactly what
        went in. Bitwise, not ``allclose`` -- this path involves no arithmetic that could
        legitimately differ.
        """
        from NSM.models.loader import load_model

        inputs = query_points()
        before = forward(reconstruction_model, inputs)

        path = save_checkpoint(reconstruction_model, tmp_path)
        reloaded = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        after = forward(reloaded, inputs)

        assert torch.equal(
            before, after
        ), f"max abs difference {(before - after).abs().max().item():.3e}"

    def test_the_comparison_can_fail(self, reconstruction_model, tmp_path):
        """
        Guard on the test above: perturb the checkpoint and confirm the forward comparison
        notices. Without this, ``torch.equal`` on two references to the same buffer would
        look like a passing round trip.

        Every float tensor is perturbed rather than one, because the VAE weights appear in
        the state dict twice -- see ``TestAliasedCheckpointEntries``.
        """
        from NSM.models.loader import load_model

        inputs = query_points()
        before = forward(reconstruction_model, inputs)

        path = save_checkpoint(reconstruction_model, tmp_path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model"] = {
            key: value + 0.01 if value.dtype.is_floating_point else value
            for key, value in checkpoint["model"].items()
        }
        torch.save(checkpoint, path)

        perturbed = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert not torch.equal(before, forward(perturbed, inputs))

    def test_load_model_returns_an_eval_mode_model(self, reconstruction_model, tmp_path):
        from NSM.models.loader import load_model

        path = save_checkpoint(reconstruction_model, tmp_path)
        assert not load_model(
            dict(ARCHITECTURE), path, model_type="triplanar", device="cpu"
        ).training

    @pytest.mark.parametrize("layout", ["model", "state_dict", "model_state_dict", None])
    def test_load_model_accepts_every_documented_checkpoint_layout(
        self, reconstruction_model, tmp_path, layout
    ):
        """
        ``loader.py:86-97`` accepts four shapes. ``save_model`` writes the first; the rest
        exist for checkpoints from elsewhere. All four must produce the same model.
        """
        from NSM.models.loader import load_model

        inputs = query_points()
        expected = forward(reconstruction_model, inputs)

        state = reconstruction_model.state_dict()
        path = str(tmp_path / f"{layout}.pth")
        torch.save({layout: state} if layout else state, path)

        loaded = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert torch.equal(expected, forward(loaded, inputs))

    def test_the_checkpoint_carries_the_epoch(self, reconstruction_model, tmp_path):
        path = save_checkpoint(reconstruction_model, tmp_path, epoch=7)
        assert torch.load(path, weights_only=False)["epoch"] == 7


class TestPaddingIsNotInTheCheckpoint:
    """
    ``TriplanarDecoder.padding`` scales query coordinates before they index the feature
    planes (``triplanar.py:322``). It is not a learned parameter, so nothing about loading
    a checkpoint constrains it: strict ``load_state_dict`` succeeds at any value and the
    model then samples at the wrong scale.

    ``load_model`` defaults it to 0.1 when the config omits it; the downstream consumer
    does not pass it at all (``steps/run_nsm.py:94-112``, 15 of 16 meaningful arguments).
    A model trained at any other value is therefore loaded wrong by both.
    """

    TRAINED_PADDING = 0.35

    @pytest.fixture(scope="class")
    def checkpoint_at_a_nondefault_padding(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("padding")
        config = dict(ARCHITECTURE, padding=self.TRAINED_PADDING)
        model = build_model(config)
        return config, model, save_checkpoint(model, directory)

    def test_a_config_without_padding_loads_without_error(self, checkpoint_at_a_nondefault_padding):
        from NSM.models.loader import load_model

        _, _, path = checkpoint_at_a_nondefault_padding
        stripped = {k: v for k, v in ARCHITECTURE.items() if k != "padding"}
        loaded = load_model(stripped, path, model_type="triplanar", device="cpu")
        assert loaded.padding == 0.1, "load_model's default changed"

    @pytest.mark.xfail(
        strict=True,
        reason="#26: padding is not in the checkpoint, so a mismatch loads silently",
    )
    def test_a_padding_mismatch_must_not_silently_change_the_sdf(
        self, checkpoint_at_a_nondefault_padding
    ):
        """
        Either the loaded model computes what the checkpoint was trained to compute, or
        loading refuses. Today it does neither: it loads cleanly and computes something
        else. The output is ``tanh``-bounded to (-1, 1), so the observed 0.063 is a large
        fraction of the full range, not a rounding artefact.
        """
        from NSM.models.loader import load_model

        _, model, path = checkpoint_at_a_nondefault_padding
        stripped = {k: v for k, v in ARCHITECTURE.items() if k != "padding"}
        loaded = load_model(stripped, path, model_type="triplanar", device="cpu")

        inputs = query_points()
        difference = (forward(model, inputs) - forward(loaded, inputs)).abs().max().item()
        assert difference < 1e-6, f"padding mismatch moved the SDF by {difference:.3e}"

    def test_stating_padding_in_the_config_restores_the_original(
        self, checkpoint_at_a_nondefault_padding
    ):
        """The fix, from the caller's side: say what you trained at."""
        from NSM.models.loader import load_model

        config, model, path = checkpoint_at_a_nondefault_padding
        loaded = load_model(config, path, model_type="triplanar", device="cpu")
        inputs = query_points()
        assert torch.equal(forward(model, inputs), forward(loaded, inputs))

    @pytest.mark.xfail(
        strict=True,
        reason="#20: normalize_coordinates still accepts a `padding` argument it never reads",
    )
    def test_normalize_coordinates_must_not_accept_a_padding_argument(self):
        """
        The fix is to DELETE the parameter, so this asserts it is gone -- not that it works.

        ``TriplanarDecoder.normalize_coordinates(self, query, plane, padding=0.1)`` accepts
        ``padding`` and divides by ``self.padding`` instead. Its sole caller,
        ``sample_plane_features``, passes none and depends on that. So making the argument
        authoritative would hand the only real caller the ``0.1`` default in place of the
        shipped 0.35 -- a measured 0.063 max SDF difference on a ``tanh``-bounded output.
        See #20's traps.

        The predecessor of this test asserted the opposite -- that ``padding`` is present
        AND that passing it changes the result -- which meant deletion reported ``xfailed``
        (green, defect looks unfixed) and honouring it reported ``XPASS(strict)`` (red,
        congratulating the harmful change). Both signals were backwards.

        ``test_self_padding_alone_governs_normalization`` below is the other half: it is a
        plain passing test, so honouring the argument turns it RED. It has to be, because
        honouring the argument ALSO makes the ``#26`` xfail above XPASS -- with
        ``self.padding`` unread, a padding mismatch really does stop changing the SDF --
        so without it the harmful fix reports as two defects fixed and nothing dissents.
        """
        model = build_model(dict(ARCHITECTURE))
        assert "padding" not in inspect.signature(model.normalize_coordinates).parameters

    def test_self_padding_alone_governs_normalization(self):
        """
        What ``normalize_coordinates`` must keep doing, whatever happens to the signature.

        Deliberately calls with the sole caller's argument list -- ``(query, plane)`` -- so
        deleting the parameter leaves this green, and reading the argument instead of
        ``self.padding`` turns it red: with no ``padding`` passed, both models below would
        then divide by the same ``0.1`` default and the outputs would coincide.

        The exact quotient is asserted, not just the inequality, because "the two differ"
        also holds for a fix that reads ``self.padding`` through some new scaling.

        ``torch.manual_seed`` is set because ``build_model`` consumes RNG during weight
        initialisation, so without it the sampled points depend on construction details.
        """
        shipped = build_model(dict(ARCHITECTURE, padding=self.TRAINED_PADDING))
        default = build_model(dict(ARCHITECTURE))
        assert (shipped.padding, default.padding) == (self.TRAINED_PADDING, 0.1)

        torch.manual_seed(0)
        points = torch.rand(8, 3)  # in [0, 1): inside the clamp at either padding

        assert not torch.equal(
            shipped.normalize_coordinates(points.clone(), "xy"),
            default.normalize_coordinates(points.clone(), "xy"),
        ), "self.padding no longer reaches normalize_coordinates"

        for model in (shipped, default):
            assert torch.equal(
                model.normalize_coordinates(points.clone(), "xy").reshape(-1, 2),
                points[:, [0, 1]] / (1 + model.padding + 10e-6),
            )


class TestSavedConfigIsEnoughToReload:
    """
    The full loop a downstream consumer performs: train, read back the
    ``model_params_config.json`` the run wrote, and rebuild from that alone.
    """

    def test_the_saved_config_rebuilds_an_identical_model(self, training_run, tmp_path):
        import json

        from NSM.models.loader import load_model

        with open(
            os.path.join(training_run["config"]["experiment_directory"], "model_params_config.json")
        ) as f:
            saved_config = json.load(f)

        inputs = query_points()
        expected = forward(training_run["model"], inputs)
        path = save_checkpoint(training_run["model"], tmp_path)

        rebuilt = load_model(saved_config, path, model_type="triplanar", device="cpu")
        assert torch.equal(expected, forward(rebuilt, inputs))

    def test_the_saved_config_records_every_architecture_key(self, training_run):
        import json

        with open(
            os.path.join(training_run["config"]["experiment_directory"], "model_params_config.json")
        ) as f:
            saved_config = json.load(f)

        missing = [key for key in ARCHITECTURE if key not in saved_config]
        assert missing == [], f"model_params_config.json omits {missing}"


class TestAliasedCheckpointEntries:
    """
    ``VAEDecoder`` registers every layer twice: once in ``self.layers`` (a ``ModuleList``)
    and again in ``self.decoder = nn.Sequential(*self.layers)`` (``triplanar.py:58-99``).
    Both are child modules, so ``state_dict()`` emits each tensor under two names.

    Loading is unaffected -- the two names alias the same parameter, so whichever is
    applied last wins and it is the same data. Two things are affected:

    * **Checkpoint size.** Every shipped NSM model on disk is roughly twice as large as its
      parameter count requires.
    * **Checkpoint surgery.** Editing a checkpoint by key -- pruning, quantizing, patching
      a layer -- silently loses the edit if only one of the two names is written. That is
      not hypothetical: it is what the first draft of ``test_the_comparison_can_fail``
      above did, and it looked like a passing round trip.
    """

    @pytest.fixture(scope="class")
    def state_dict(self):
        return build_model(dict(ARCHITECTURE)).state_dict()

    @pytest.mark.xfail(strict=True, reason="#27: VAEDecoder registers every layer twice")
    def test_each_parameter_must_appear_once_in_the_state_dict(self, state_dict):
        aliased = {
            name
            for name in state_dict
            if name.startswith("vae_decoder.layers.")
            and name.replace("layers.", "decoder.") in state_dict
        }
        assert (
            not aliased
        ), f"{len(aliased)} tensors are stored under two names: {sorted(aliased)[:3]}"

    @pytest.mark.xfail(strict=True, reason="#27: aliasing inflates every checkpoint ~1.92x")
    def test_the_checkpoint_must_not_hold_more_elements_than_the_model_has_parameters(
        self, state_dict
    ):
        model = build_model(dict(ARCHITECTURE))
        stored = sum(tensor.numel() for tensor in state_dict.values())
        parameters = sum(p.numel() for p in model.parameters())
        assert stored == parameters, f"{stored} elements stored for {parameters} parameters"

    @pytest.mark.xfail(strict=True, reason="#27: an edit to one alias is reverted by the other")
    def test_editing_a_checkpoint_by_key_must_take_effect(self, tmp_path):
        """The failure mode, demonstrated."""
        from NSM.models.loader import load_model

        model = build_model(dict(ARCHITECTURE))
        inputs = query_points()
        before = forward(model, inputs)

        path = save_checkpoint(model, tmp_path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model"]["vae_decoder.layers.0.weight"] = (
            checkpoint["model"]["vae_decoder.layers.0.weight"] + 1.0
        )
        torch.save(checkpoint, path)

        edited = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert not torch.equal(
            before, forward(edited, inputs)
        ), "the +1.0 written to vae_decoder.layers.0.weight had no effect on the model"


class TestModelConfigMapping:
    """
    ``build_model`` reaches into ``loader._get_triplanar_params`` because NSM has no public
    "build the model this config describes" call. These tests pin what that private
    function is being relied on for, so the coupling is visible if the decoder registry
    changes it.
    """

    def test_the_harness_builds_what_load_model_would_build(self, tmp_path):
        from NSM.models.loader import load_model

        model = build_model(dict(ARCHITECTURE))
        path = save_checkpoint(model, tmp_path)
        loaded = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")

        assert type(loaded) is type(model)
        assert loaded.padding == model.padding
        assert loaded.sdf_latent_size == model.sdf_latent_size
        assert loaded.n_objects == model.n_objects

    def test_the_config_keys_the_triplanar_branch_reads(self):
        """
        Recorded as data: the mapping from config key to constructor argument, which the
        consumer duplicates by hand. If a key is renamed, this is where it shows up.
        """
        from NSM.models.loader import _get_triplanar_params

        _, params = _get_triplanar_params(dict(ARCHITECTURE))
        assert params["latent_dim"] == ARCHITECTURE["latent_size"]
        assert params["n_objects"] == ARCHITECTURE["objects_per_decoder"]
        assert params["sdf_weight_norm"] == ARCHITECTURE["weight_norm"]
        assert params["sdf_final_activation"] == ARCHITECTURE["final_activation"]
        assert params["sum_sdf_features"] == ARCHITECTURE["sum_conv_output_features"]
        assert params["padding"] == ARCHITECTURE["padding"]

    def test_mesh_names_is_not_a_constructor_argument(self):
        """
        ``mesh_names`` exists in the config for exactly the surface-identity problem
        ``reconstruct_mesh``'s positional ``mesh`` list has, and no model ever sees it.
        """
        from NSM.models.loader import _get_triplanar_params

        _, params = _get_triplanar_params(dict(ARCHITECTURE, mesh_names=["bone", "cart"]))
        assert "mesh_names" not in params
