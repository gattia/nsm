"""
Test module for NSM model loader functionality.

Tests the load_model function and related utilities for loading
pre-trained Neural Shape Models from configuration and state files.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from NSM.models import (
    Decoder,
    TriplanarDecoder,
    get_model_config_template,
    list_supported_models,
    load_model,
)


class TestModelLoader:
    """Test class for model loader functionality."""

    def test_list_supported_models(self):
        """Test that supported models list is returned correctly."""
        models = list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        expected_models = ["triplanar", "deepsdf"]
        for model in expected_models:
            assert model in models

    def test_get_model_config_template_all_types(self):
        """Test getting config templates for all supported model types."""
        models = list_supported_models()

        for model_type in models:
            config = get_model_config_template(model_type)
            assert isinstance(config, dict)
            assert len(config) > 0

            # Each config should have some required parameters
            assert "latent_size" in config

    def test_get_model_config_template_invalid_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_config_template("invalid_model_type")

    def test_model_initialization_from_templates(self):
        """Test that all models can be initialized using their config templates."""
        models = list_supported_models()

        for model_type in models:
            config = get_model_config_template(model_type)

            # Import the internal parameter extraction functions
            from NSM.models.loader import _get_deepsdf_params, _get_triplanar_params

            if model_type == "triplanar":
                model_class, params = _get_triplanar_params(config)
                assert model_class == TriplanarDecoder
            elif model_type == "deepsdf":
                model_class, params = _get_deepsdf_params(config)
                assert model_class == Decoder

            # Initialize the model
            model = model_class(**params)
            assert isinstance(model, torch.nn.Module)

    def test_load_model_invalid_type(self):
        """Test that load_model raises ValueError for invalid model type."""
        config = get_model_config_template("triplanar")

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            # Create dummy state dict
            dummy_state = {"model": {}}
            torch.save(dummy_state, tmp_file.name)

            with pytest.raises(ValueError, match="Unknown model type"):
                load_model(config, tmp_file.name, model_type="invalid_type")

    def test_load_model_missing_file(self):
        """Test that load_model raises FileNotFoundError for missing file."""
        config = get_model_config_template("triplanar")

        with pytest.raises(FileNotFoundError):
            load_model(config, "/nonexistent/path/model.pt", model_type="triplanar")

    def test_load_model_missing_config_keys(self):
        """Test that load_model raises KeyError for missing required config keys."""
        # Empty config should fail
        empty_config = {}

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            dummy_state = {"model": {}}
            torch.save(dummy_state, tmp_file.name)

            with pytest.raises(KeyError, match="Missing required configuration keys"):
                load_model(empty_config, tmp_file.name, model_type="triplanar")


@pytest.fixture
def temp_model_files():
    """Create temporary model files with proper state dicts for testing."""
    models_data = {}

    # Create a simple model for each type and save its state
    for model_type in list_supported_models():
        config = get_model_config_template(model_type)

        # Modify configs to be smaller for faster testing
        if model_type == "triplanar":
            config["latent_size"] = 64
            config["conv_hidden_dims"] = [128, 128]
            config["sdf_hidden_dims"] = [128, 128]
            config["sdf_latent_size"] = 32
            model = TriplanarDecoder(
                **{
                    "latent_dim": config["latent_size"],
                    "n_objects": config["objects_per_decoder"],
                    "conv_hidden_dims": config["conv_hidden_dims"],
                    "conv_deep_image_size": config["conv_deep_image_size"],
                    "conv_norm": config["conv_norm"],
                    "conv_norm_type": config["conv_norm_type"],
                    "conv_start_with_mlp": config["conv_start_with_mlp"],
                    "sdf_latent_size": config["sdf_latent_size"],
                    "sdf_hidden_dims": config["sdf_hidden_dims"],
                    "sdf_weight_norm": config["weight_norm"],
                    "sdf_final_activation": config["final_activation"],
                    "sdf_activation": config["activation"],
                    "sdf_dropout_prob": config["dropout_prob"],
                    "sum_sdf_features": config["sum_conv_output_features"],
                    "conv_pred_sdf": config["conv_pred_sdf"],
                    "padding": config["padding"],
                }
            )

        elif model_type == "deepsdf":
            config["latent_size"] = 64
            config["layer_dimensions"] = [128, 128, 128]
            model = Decoder(
                latent_size=config["latent_size"],
                dims=config["layer_dimensions"],
                n_objects=config["objects_per_decoder"],
                dropout=config["layers_with_dropout"],
                dropout_prob=config["dropout_prob"],
                latent_in=config["layer_latent_in"],
                weight_norm=config["weight_norm"],
                activation=config["activation"],
                final_activation=config["final_activation"],
                concat_latent_input=config["concat_latent_input"],
                progressive_add_depth=config["progressive_add_depth"],
                layer_split=config["layer_split"],
            )

        # Save model to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, temp_file.name)
        temp_file.close()

        models_data[model_type] = {
            "config": config,
            "file_path": temp_file.name,
            "original_model": model,
        }

    yield models_data

    # Cleanup
    for data in models_data.values():
        if os.path.exists(data["file_path"]):
            os.unlink(data["file_path"])


class TestModelLoadingFullWorkflow:
    """Test class for complete model loading workflow with actual models."""

    def test_load_model_full_workflow(self, temp_model_files):
        """Test complete workflow of loading models from saved states."""
        for model_type, data in temp_model_files.items():
            config = data["config"]
            file_path = data["file_path"]
            original_model = data["original_model"]

            # Load the model using our loader
            loaded_model = load_model(config, file_path, model_type=model_type, device="cpu")

            # Verify the loaded model
            assert isinstance(loaded_model, torch.nn.Module)
            assert type(loaded_model) == type(original_model)
            assert not loaded_model.training  # Should be in eval mode

            # Test that the model can perform inference
            latent_size = config["latent_size"]
            batch_size = 10

            # Create test input: [latent, xyz]
            test_input = torch.randn(batch_size, latent_size + 3)

            with torch.no_grad():
                output = loaded_model(test_input)

            assert output.shape[0] == batch_size
            assert output.shape[1] == config.get("objects_per_decoder", 1)

    def test_different_state_dict_formats(self, temp_model_files):
        """Test loading models with different state dict save formats."""
        # Test with triplanar model
        data = temp_model_files["triplanar"]
        config = data["config"]
        original_model = data["original_model"]

        # Test different save formats
        formats = [
            {"model": original_model.state_dict()},
            {"state_dict": original_model.state_dict()},
            {"model_state_dict": original_model.state_dict()},
            original_model.state_dict(),  # Direct state dict
        ]

        for i, state_format in enumerate(formats):
            with tempfile.NamedTemporaryFile(suffix=f"_format_{i}.pt", delete=False) as tmp_file:
                torch.save(state_format, tmp_file.name)

                try:
                    # Should load successfully regardless of format
                    loaded_model = load_model(
                        config, tmp_file.name, model_type="triplanar", device="cpu"
                    )
                    assert isinstance(loaded_model, TriplanarDecoder)
                    assert not loaded_model.training
                finally:
                    os.unlink(tmp_file.name)

    def test_device_handling(self, temp_model_files):
        """Test that models are loaded to the correct device."""
        data = temp_model_files["deepsdf"]
        config = data["config"]
        file_path = data["file_path"]

        # Test loading to CPU
        model_cpu = load_model(config, file_path, model_type="deepsdf", device="cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Test automatic device detection (should default to CPU in test environment)
        model_auto = load_model(config, file_path, model_type="deepsdf", device=None)
        assert next(model_auto.parameters()).device.type in ["cpu", "cuda"]


class TestConvNormTypeMustBeStated:
    """
    ``conv_norm_type`` decides the VAE's normalization, and until Aug 2026 four places
    defaulted it and disagreed: ``"batch"`` in ``VAEDecoder``, ``TriplanarDecoder``,
    ``_get_triplanar_params`` and the triplanar template; ``"layer"`` in the (since
    removed, SCOPE.md section 2.9) two_stage loader branch and defaults, and in
    ``NSM/configs/default_config.json``.

    **The value nothing has ever trained was the one that won three of those.** Every
    ShapeMedKnee config -- 647, 551, the 2024 training config and the regenerated
    ``default_config.json`` -- says ``"layer"``. And ``"layer"`` is not cosmetic: it is the
    only thing making the VAE nonlinear at all, because the pointwise activation was never
    wired in (``ARCHITECTURE.md`` section 7.1). Under ``"batch"`` the stack trains nonlinear
    (batch statistics couple samples) and evaluates affine (running statistics).

    A mismatch against a checkpoint does not load silently -- ``BatchNorm2d`` and
    ``LayerNorm`` differ in both key set and shape, so torch refuses. What the silent
    default cost was a *fresh* run started from the template, which inherited a
    configuration nobody has trained and nothing would flag.
    """

    def test_a_triplanar_config_without_it_is_refused(self):
        stripped = {
            k: v for k, v in get_model_config_template("triplanar").items() if k != "conv_norm_type"
        }
        with pytest.raises(KeyError, match="conv_norm_type"):
            load_model(stripped, "/nonexistent.pt", model_type="triplanar")

    def test_the_triplanar_template_advertises_the_value_that_was_trained(self):
        """
        A template is what a NEW config should look like, so it must not hand someone the
        configuration nothing has been trained with.
        """
        assert get_model_config_template("triplanar")["conv_norm_type"] == "layer"

    def test_a_triplanar_config_without_conv_activation_is_refused(self):
        """
        Same contract, different key: ``conv_activation`` decides the module *layout*, so a
        config that does not state it does not describe an architecture. ``null`` is the
        historical stack -- see ``TestTheOptInConvActivation``.
        """
        stripped = {
            k: v
            for k, v in get_model_config_template("triplanar").items()
            if k != "conv_activation"
        }
        with pytest.raises(KeyError, match="conv_activation"):
            load_model(stripped, "/nonexistent.pt", model_type="triplanar")

    def test_the_triplanar_template_defaults_to_the_historical_architecture(self):
        """``null``, not an activation: a template that flipped the architecture would make
        every checkpoint anyone owns unloadable from it."""
        assert get_model_config_template("triplanar")["conv_activation"] is None

    def test_the_loader_keeps_no_silent_default_for_it(self):
        """
        The structural half, and the one that stops this regressing: a fix that only
        aligned the two literals would leave the next person free to add a third
        ``config.get("conv_norm_type", ...)``. There must be no default to disagree about.
        """
        import ast
        import inspect

        import NSM.models.loader as loader

        tree = ast.parse(inspect.getsource(loader))
        defaulted = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("conv_norm_type", "conv_activation")
        ]
        assert defaulted == [], f"{len(defaulted)} silent default(s) for an architecture key"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
