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
        Bitwise, since no arithmetic on this path can legitimately differ. Through
        ``load_model`` (in eval mode, epoch recorded), through each checkpoint layout it
        accepts, and through the consumer's bare ``load_state_dict`` (``SCOPE`` §4).
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
        """Perturbing every float tensor must show, or ``torch.equal`` proves nothing."""
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
        """The consumer's loop: train, read back ``model_params_config.json``, rebuild."""
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
        The refusal is worth its cost: a mismatch moves the ``tanh``-bounded SDF by more
        than 1e-2. Asserted as a floor, so padding mattering less would show here.
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
        ``normalize_coordinates`` took a ``padding`` argument and divided by ``self.padding``.
        Its one caller passes none, so honouring the argument would have handed it the 0.1
        default instead of the shipped 0.35: a 0.063 SDF difference. The argument was
        deleted. Called with the caller's own arguments, the exact quotient is asserted.
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
    ``VAEDecoder`` registered every layer twice until #27, so each tensor was saved under two
    names and a by-key edit to one was reverted by the other. Shipped checkpoints still carry
    both, so a load-time hook drops the ``layers.*`` aliases, and where they disagree the
    ``decoder.*`` value wins, as it always did.
    """

    def test_each_parameter_is_saved_once_and_an_edit_takes_effect(self, tmp_path):
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
