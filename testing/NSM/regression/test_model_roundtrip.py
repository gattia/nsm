"""
Checkpoint round trip: ``save_model`` then ``load_model`` computes exactly what went in.

And ``padding``, which is not a learned parameter: a checkpoint trained at one value loads
cleanly at another and then samples the feature planes at the wrong scale.
"""

import inspect
import json
import os

import pytest
import torch
from _harness import ARCHITECTURE, LATENT_SIZE, build_model


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
    def test_a_trained_model_round_trips_bitwise_through_every_path(
        self, reconstruction_model, tmp_path
    ):
        """
        Fails if a checkpoint reloaded by ``load_model``, in any of the four layouts it
        accepts, or by a bare ``load_state_dict`` computes a different output, ``load_model``
        returns train mode, or ``save_model`` stops recording ``epoch``.

        Bitwise, since no arithmetic on this path can legitimately differ. The bare
        ``load_state_dict`` is the consumer's path (``SCOPE`` §4).
        """
        from NSM.models.loader import load_model

        inputs = query_points()
        expected = forward(reconstruction_model, inputs)
        path = save_checkpoint(reconstruction_model, tmp_path, epoch=7)
        assert torch.load(path, weights_only=False)["epoch"] == 7

        loaded = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert not loaded.training
        assert torch.equal(expected, forward(loaded, inputs))

        rebuilt = build_model(dict(ARCHITECTURE))
        rebuilt.load_state_dict(torch.load(path, weights_only=False)["model"])
        assert torch.equal(expected, forward(rebuilt, inputs))

        state = reconstruction_model.state_dict()
        for layout in ("model", "state_dict", "model_state_dict", None):
            layout_path = str(tmp_path / f"{layout}.pth")
            torch.save({layout: state} if layout else state, layout_path)
            loaded = load_model(
                dict(ARCHITECTURE), layout_path, model_type="triplanar", device="cpu"
            )
            assert torch.equal(expected, forward(loaded, inputs)), layout

    def test_the_comparison_can_fail(self, reconstruction_model, tmp_path):
        """
        Fails if a checkpoint with every float tensor shifted by 0.01 loads through
        ``load_model`` to the same output, which would make the bitwise round trip vacuous.
        """
        from NSM.models.loader import load_model

        inputs = query_points()
        path = save_checkpoint(reconstruction_model, tmp_path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model"] = {
            key: value + 0.01 if value.dtype.is_floating_point else value
            for key, value in checkpoint["model"].items()
        }
        torch.save(checkpoint, path)
        perturbed = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert not torch.equal(forward(reconstruction_model, inputs), forward(perturbed, inputs))

    def test_the_config_a_run_saves_is_enough_to_reload_it(self, training_run, tmp_path):
        """
        Fails if ``train_deep_sdf``'s ``model_params_config.json`` omits an ``ARCHITECTURE``
        key, or records a value from which ``load_model`` builds a different model.

        The consumer's loop: train, read back the config, rebuild. The trained model is
        re-saved with ``save_model`` rather than loaded from the run's own checkpoint.
        """
        from NSM.models.loader import load_model

        directory = training_run["config"]["experiment_directory"]
        with open(os.path.join(directory, "model_params_config.json"), encoding="utf-8") as f:
            saved_config = json.load(f)
        assert [key for key in ARCHITECTURE if key not in saved_config] == []

        inputs = query_points()
        path = save_checkpoint(training_run["model"], tmp_path)
        rebuilt = load_model(saved_config, path, model_type="triplanar", device="cpu")
        assert torch.equal(forward(training_run["model"], inputs), forward(rebuilt, inputs))


class TestPaddingIsNotInTheCheckpoint:
    """
    ``TriplanarDecoder.padding`` scales coordinates before they index the feature planes. A
    strict ``load_state_dict`` succeeds at any value, so ``load_model`` requires the config
    to state it (#26) and says what to write.
    """

    TRAINED_PADDING = 0.35

    def test_a_config_must_state_it_and_stating_it_restores_the_model(self, tmp_path):
        """
        Fails if ``load_model`` accepts a triplanar config without ``padding``, leaves the
        0.1 repair out of its ``KeyError``, or builds a model whose output does not follow
        the stated ``padding`` (#26; KNOWN_ISSUES History 16).

        The refusal is worth its cost: a mismatch (0.35 trained, 0.1 loaded) moves the
        ``tanh``-bounded SDF by more than 1e-2. Asserted as a floor, so padding mattering
        less would show here.
        """
        from NSM.models.loader import load_model

        config = dict(ARCHITECTURE, padding=self.TRAINED_PADDING)
        model = build_model(config)
        path = save_checkpoint(model, tmp_path)

        stripped = {k: v for k, v in ARCHITECTURE.items() if k != "padding"}
        with pytest.raises(KeyError) as raised:
            load_model(stripped, path, model_type="triplanar", device="cpu")
        assert "padding" in str(raised.value)
        assert "0.1" in str(raised.value) and "not a learned parameter" in str(raised.value)

        inputs = query_points()
        loaded = load_model(config, path, model_type="triplanar", device="cpu")
        assert torch.equal(forward(model, inputs), forward(loaded, inputs))
        mismatched = load_model(
            dict(config, padding=0.1), path, model_type="triplanar", device="cpu"
        )
        assert (forward(model, inputs) - forward(mismatched, inputs)).abs().max() > 1e-2

    def test_self_padding_alone_governs_normalization(self):
        """
        Fails if ``TriplanarDecoder.normalize_coordinates`` takes a ``padding`` argument, or
        its result is not exactly ``xy / (1 + self.padding + 10e-6)`` at 0.35 and at 0.1.

        Its one caller passes no ``padding``, so an argument defaulting to 0.1 would replace
        a 0.35 model's own value: a 0.063 SDF difference. It is called here with the
        caller's own arguments.
        """
        shipped = build_model(dict(ARCHITECTURE, padding=self.TRAINED_PADDING))
        default = build_model(dict(ARCHITECTURE))
        assert "padding" not in inspect.signature(shipped.normalize_coordinates).parameters

        torch.manual_seed(0)
        points = torch.rand(8, 3)
        for model in (shipped, default):
            assert torch.equal(
                model.normalize_coordinates(points.clone(), "xy").reshape(-1, 2),
                points[:, [0, 1]] / (1 + model.padding + 10e-6),
            )


class TestAliasedCheckpointEntries:
    """
    A checkpoint saved before #27 carries every ``VAEDecoder`` tensor twice, under
    ``decoder.*`` and a ``layers.*`` alias. The shipped checkpoints are among them. A
    load-time hook drops the aliases, so where the two disagree the ``decoder.*`` value wins.
    """

    def test_each_parameter_is_saved_once_and_an_edit_takes_effect(self, tmp_path):
        """
        Fails if ``TriplanarDecoder.state_dict()`` carries ``vae_decoder.layers.*`` aliases
        or any tensor twice, or a by-key edit to ``vae_decoder.decoder.*`` does not change
        the loaded model's output (#27).

        The element-count check assumes the model has no buffers: true under layer norm,
        false under batch norm.
        """
        from NSM.models.loader import load_model

        model = build_model(dict(ARCHITECTURE))
        state = model.state_dict()
        assert not [name for name in state if name.startswith("vae_decoder.layers.")]
        assert sum(t.numel() for t in state.values()) == sum(p.numel() for p in model.parameters())

        inputs = query_points()
        path = save_checkpoint(model, tmp_path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model"]["vae_decoder.decoder.0.weight"] += 1.0
        torch.save(checkpoint, path)
        edited = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert not torch.equal(forward(model, inputs), forward(edited, inputs))

    def test_a_pre_fix_checkpoint_still_loads_both_ways(self, tmp_path):
        """
        Fails if a checkpoint carrying ``vae_decoder.layers.*`` aliases fails a strict
        ``load_state_dict`` or ``load_model``, or loads the alias values over ``decoder.*``
        (#27).

        The aliases are offset by 1.0, so a load that let them win changes the output.
        kneepipeline loads the shipped checkpoints with a strict ``load_state_dict``.
        """
        from NSM.models.loader import load_model

        model = build_model(dict(ARCHITECTURE))
        inputs = query_points()
        expected = forward(model, inputs)
        old_state = dict(model.state_dict())
        for key, value in model.state_dict().items():
            if key.startswith("vae_decoder.decoder."):
                alias = key.replace(".decoder.", ".layers.", 1)
                old_state[alias] = value + 1.0 if value.dtype.is_floating_point else value

        bare = build_model(dict(ARCHITECTURE))
        bare.load_state_dict(old_state, strict=True)
        assert torch.equal(expected, forward(bare, inputs))

        path = str(tmp_path / "old_format.pth")
        torch.save({"model": old_state}, path)
        via_loader = load_model(dict(ARCHITECTURE), path, model_type="triplanar", device="cpu")
        assert torch.equal(expected, forward(via_loader, inputs))
