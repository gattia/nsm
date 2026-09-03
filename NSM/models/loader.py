"""
Model loader utilities for Neural Shape Models (NSM).

This module provides functions to load pre-trained NSM models from configuration
and state files, supporting multiple model architectures.
"""

import json
import warnings
from typing import Any, Dict, Optional, Union

import torch

from .deep_sdf import Decoder
from .triplanar import TriplanarDecoder
from .two_stage import TwoStageDecoder


def load_model(
    config: Dict[str, Any],
    path_model_state: str,
    model_type: str = "triplanar",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """
    Loads a pre-trained Neural Shape Model (NSM) from configuration and state files.

    Supports 'triplanar', 'deepsdf', and 'two_stage' model architectures.
    Initializes the model based on parameters in the `config` dictionary, loads the
    learned weights from `path_model_state`, moves the model to the specified device,
    and sets it to evaluation mode.

    Args:
        config (Dict[str, Any]): A dictionary containing model configuration parameters
            (e.g., latent_size, layer_dimensions, activation functions).
        path_model_state (str): Path to the .pt or .pth file containing the
            saved model state_dict.
        model_type (str, optional): The type of NSM architecture to load.
            Supported values are 'triplanar', 'deepsdf', and 'two_stage'.
            Defaults to 'triplanar'.
        device (str or torch.device, optional): Device to load the model on.
            If None, defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        torch.nn.Module: The loaded and initialized NSM model, ready for evaluation.

    Raises:
        ValueError: If `model_type` is not one of the supported values.
        FileNotFoundError: If `path_model_state` does not exist.
        KeyError: If required configuration parameters are missing.
    """

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model class and parameters based on model type
    if model_type == "triplanar":
        model_class, params = _get_triplanar_params(config)
    elif model_type == "deepsdf":
        model_class, params = _get_deepsdf_params(config)
    elif model_type == "two_stage":
        model_class, params = _get_two_stage_params(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: triplanar, deepsdf, two_stage"
        )

    # Initialize model
    try:
        model = model_class(**params)
    except Exception as e:
        raise ValueError(f"Failed to initialize {model_type} model with provided config: {e}")

    # Load model state
    try:
        saved_model_state = torch.load(path_model_state, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state file not found: {path_model_state}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model state from {path_model_state}: {e}")

    # Handle different save formats
    if isinstance(saved_model_state, dict):
        if "model" in saved_model_state:
            state_dict = saved_model_state["model"]
        elif "state_dict" in saved_model_state:
            state_dict = saved_model_state["state_dict"]
        elif "model_state_dict" in saved_model_state:
            state_dict = saved_model_state["model_state_dict"]
        else:
            # Assume the dict itself is the state_dict
            state_dict = saved_model_state
    else:
        state_dict = saved_model_state

    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {e}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


class MissingArchitectureKeys(KeyError):
    """
    A ``KeyError`` whose message survives being printed.

    ``KeyError.__str__`` is ``repr(args[0])``, so a multi-line message renders with
    literal ``\n`` escapes and a JSON block in it arrives unusable -- which is the whole
    point of the message below. Subclassing ``KeyError`` rather than raising something
    else keeps every ``except KeyError`` that already exists working.
    """

    def __str__(self):
        return self.args[0]


#: ``key -> (what it decides, the value a model trained before Aug 2026 ran at)``. Both
#: halves go into the refusal below, so this table is the message.
#:
#: Issues #26 (``padding``) and #45 (``conv_norm_type``); ``conv_activation`` is the
#: activation the VAE built and never appended, ``docs/ARCHITECTURE.md`` section 7.1.
REQUIRED_ARCHITECTURE_KEYS = {
    "padding": (
        "scales query coordinates before they index the feature planes and is not a "
        "learned parameter, so a checkpoint loads clean at the wrong value and then "
        "samples at the wrong scale, silently",
        0.1,
    ),
    "conv_activation": (
        "decides the architecture rather than a hyperparameter: null is the historical "
        "stack, which has NO pointwise activation because one was built and never "
        "appended until Aug 2026. Any other value renumbers every later state-dict key",
        None,
    ),
    "conv_norm_type": (
        'decides the VAE normalization. "layer" is what every ShapeMedKnee model and the '
        'shipped default_config.json trained with; "batch" was the constructor default '
        "until v0.3.0 and adds running statistics, so a wrong guess fails at load",
        "layer",
    ),
}


def _refuse_missing_architecture_keys(config, keys, model_type):
    """
    Refuse a config missing any of ``keys``, naming **all** of them in one message.

    One message rather than one per key, because these used to be separate ``raise``
    statements and repairing a pre-Aug-2026 config cost one round-trip per key. The
    message ends with a JSON object that repairs the config in a single edit;
    ``testing/NSM/models/test_config_repair.py`` parses that object and applies it, so it
    cannot drift from what the code requires.
    """
    missing = [key for key in keys if key not in config]
    if not missing:
        return

    plural = "key" if len(missing) == 1 else "keys"
    explained = "\n".join(f"  {key}: {REQUIRED_ARCHITECTURE_KEYS[key][0]}." for key in missing)
    historical = json.dumps({key: REQUIRED_ARCHITECTURE_KEYS[key][1] for key in missing}, indent=4)
    raise MissingArchitectureKeys(
        f"This {model_type} config does not state {len(missing)} architecture {plural} "
        f"that cannot be recovered from the checkpoint: {', '.join(missing)}.\n\n"
        f"{explained}\n\n"
        "Configs written before Aug 2026 omit these, and every model trained before then "
        "ran at the values below. Add them to the config in one edit -- or state what you "
        "actually trained with, if it was not this:\n\n"
        f"{historical}\n\n"
        "Background: docs/ARCHITECTURE.md section 7.1, docs/KNOWN_ISSUES.md, "
        "and issues #26 and #45."
    )


def _get_triplanar_params(config: Dict[str, Any]) -> tuple:
    """Extract TriplanarDecoder parameters from config."""
    required_keys = ["latent_size"]
    _check_required_keys(config, required_keys, "triplanar")

    _refuse_missing_architecture_keys(config, REQUIRED_ARCHITECTURE_KEYS, "triplanar")

    params = {
        "latent_dim": config["latent_size"],
        "n_objects": config.get("objects_per_decoder", 1),
        "conv_hidden_dims": config.get("conv_hidden_dims", [512, 512, 512, 512, 512]),
        "conv_deep_image_size": config.get("conv_deep_image_size", 2),
        "conv_norm": config.get("conv_norm", True),
        "conv_norm_type": config["conv_norm_type"],
        "conv_start_with_mlp": config.get("conv_start_with_mlp", True),
        "conv_activation": config["conv_activation"],
        "sdf_latent_size": config.get("sdf_latent_size", 128),
        "sdf_hidden_dims": config.get("sdf_hidden_dims", [512, 512, 512]),
        "sdf_weight_norm": config.get("weight_norm", True),
        "sdf_final_activation": config.get("final_activation", "tanh"),
        "sdf_activation": config.get("activation", "relu"),
        "sdf_dropout_prob": config.get("dropout_prob", 0.0),
        "sum_sdf_features": config.get("sum_conv_output_features", True),
        "conv_pred_sdf": config.get("conv_pred_sdf", False),
        # Required above, so there is no default to fall back to.
        "padding": config["padding"],
    }

    return TriplanarDecoder, params


def _get_deepsdf_params(config: Dict[str, Any]) -> tuple:
    """Extract Decoder (DeepSDF) parameters from config."""
    required_keys = ["latent_size", "layer_dimensions"]
    _check_required_keys(config, required_keys, "deepsdf")

    # Handle deprecated parameters
    if "latent_dropout" in config:
        warnings.warn(
            "latent_dropout is deprecated and ignored: the latent-input dropout it once "
            "enabled no longer exists, and per-layer dropout (layers_with_dropout + "
            "dropout_prob) is a different mechanism, not a replacement. Delete the key.",
            DeprecationWarning,
        )

    params = {
        "latent_size": config["latent_size"],
        "dims": config["layer_dimensions"],
        "n_objects": config.get("objects_per_decoder", 1),
        "dropout": config.get("layers_with_dropout", None),
        "dropout_prob": config.get("dropout_prob", 0.2),
        # norm_layers, xyz_in_all and latent_noise_sigma are deleted arguments, still
        # mapped so a config that sets one reaches Decoder's refusal instead of being
        # dropped here. Permanent: configs written before the removal exist forever.
        "norm_layers": config.get("layers_with_norm", ()),
        "latent_in": config.get("layer_latent_in", ()),
        "weight_norm": config.get("weight_norm", True),
        "xyz_in_all": config.get("xyz_in_all", None),
        "activation": config.get("activation", "relu"),
        "final_activation": config.get("final_activation", "tanh"),
        "concat_latent_input": config.get("concat_latent_input", False),
        "progressive_add_depth": config.get("progressive_add_depth", False),
        "layer_split": config.get("layer_split", None),
        "latent_noise_sigma": config.get("latent_noise_sigma", None),
    }

    return Decoder, params


def _get_two_stage_params(config: Dict[str, Any]) -> tuple:
    """Extract TwoStageDecoder parameters from config."""
    required_keys = ["latent_size"]
    _check_required_keys(config, required_keys, "two_stage")

    # Extract triplanar and MLP specific parameters
    triplanar_params = {}
    mlp_params = {}

    # Triplanar parameters. Either branch builds the same TriplanarDecoder as the
    # triplanar model type, so both are held to the same required keys -- including
    # `padding`, which this branch used to drop even when the config stated it.
    if "triplanar_params" in config:
        triplanar_params = config["triplanar_params"].copy()
        _refuse_missing_architecture_keys(
            triplanar_params, REQUIRED_ARCHITECTURE_KEYS, "two_stage triplanar_params"
        )
    else:
        _refuse_missing_architecture_keys(config, REQUIRED_ARCHITECTURE_KEYS, "two_stage")
        triplanar_params = {
            "conv_hidden_dims": config.get("conv_hidden_dims", [512, 512, 512, 512, 512]),
            "conv_deep_image_size": config.get("conv_deep_image_size", 2),
            "conv_norm": config.get("conv_norm", True),
            "conv_norm_type": config["conv_norm_type"],
            "conv_start_with_mlp": config.get("conv_start_with_mlp", True),
            "conv_activation": config["conv_activation"],
            "sdf_latent_size": config.get("sdf_latent_size", 128),
            "sdf_hidden_dims": config.get("sdf_hidden_dims", [512, 512, 512]),
            "sdf_weight_norm": config.get("weight_norm", True),
            "sdf_final_activation": config.get("final_activation", "tanh"),
            "sdf_activation": config.get("activation", "relu"),
            "padding": config["padding"],
        }

    # MLP parameters
    if "mlp_params" in config:
        mlp_params = config["mlp_params"].copy()
    else:
        # Use default MLP params with config overrides
        mlp_params = {
            "dims": list(config.get("layer_dimensions", (512, 512, 512, 512, 512, 512, 512, 512))),
            "dropout": config.get("layers_with_dropout", None),
            "dropout_prob": config.get("dropout_prob", 0.0),
            "norm_layers": config.get("layers_with_norm", ()),
            "latent_in": config.get("layer_latent_in", ()),
            "weight_norm": config.get("weight_norm", True),
            "xyz_in_all": config.get("xyz_in_all", None),
            "activation": config.get("activation", "relu"),
            "final_activation": config.get("final_activation", "tanh"),
            "concat_latent_input": config.get("concat_latent_input", True),
        }

    params = {
        "latent_size": config["latent_size"],
        "n_objects": config.get("objects_per_decoder", 2),
        "triplanar_params": triplanar_params,
        "mlp_params": mlp_params,
    }

    return TwoStageDecoder, params


def _check_required_keys(config: Dict[str, Any], required_keys: list, model_type: str):
    """Check that all required keys are present in config."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(
            f"Missing required configuration keys for {model_type} model: {missing_keys}"
        )


def list_supported_models() -> list:
    """Return a list of supported model types."""
    return ["triplanar", "deepsdf", "two_stage"]


def get_model_config_template(model_type: str) -> Dict[str, Any]:
    """
    Get a template configuration dictionary for a specific model type.

    Args:
        model_type (str): The model type to get template for.

    Returns:
        Dict[str, Any]: Template configuration with default values and descriptions.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "triplanar":
        return {
            # Required
            "latent_size": 256,
            # Optional
            "objects_per_decoder": 1,
            "conv_hidden_dims": [512, 512, 512, 512, 512],
            "conv_deep_image_size": 2,
            "conv_norm": True,
            # 'layer', not the constructor's 'batch': "layer" is what every ShapeMedKnee
            # model and NSM's own default_config.json were trained with, and "batch" makes
            # the VAE train nonlinear (batch statistics couple samples) and evaluate affine
            # (running statistics) -- a different function class fitted and deployed. See
            # docs/ARCHITECTURE.md section 7.1.
            "conv_norm_type": "layer",  # 'batch' or 'layer'
            "conv_start_with_mlp": True,
            # null is the HISTORICAL architecture -- no pointwise activation in the conv
            # stack, which is what every model trained before Aug 2026 is. Any other value
            # ('relu', 'leaky_relu', 'swish', ...) builds a layout no existing checkpoint
            # fits. See docs/ARCHITECTURE.md section 7.1 and NSM_TRAINING_IDEAS Idea 13.
            "conv_activation": None,
            "sdf_latent_size": 128,
            "sdf_hidden_dims": [512, 512, 512],
            "weight_norm": True,
            "final_activation": "tanh",  # 'tanh', 'sigmoid', 'linear'
            "activation": "relu",  # 'relu', 'leaky_relu', 'sin', etc.
            "dropout_prob": 0.0,
            "sum_conv_output_features": True,
            "conv_pred_sdf": False,
            "padding": 0.1,
        }

    elif model_type == "deepsdf":
        return {
            # Required
            "latent_size": 256,
            "layer_dimensions": [512, 512, 512, 512, 512, 512, 512, 512],
            # Optional
            "objects_per_decoder": 1,
            "layers_with_dropout": None,  # List of layer indices or None
            "dropout_prob": 0.2,
            "layer_latent_in": (),  # Tuple of layer indices
            "weight_norm": True,
            "activation": "relu",
            "final_activation": "tanh",
            "concat_latent_input": False,
            "progressive_add_depth": False,
            "layer_split": None,
        }

    elif model_type == "two_stage":
        return {
            # Required
            "latent_size": 512,
            # Optional
            "objects_per_decoder": 2,
            # Can specify nested params or use top-level params
            "triplanar_params": {
                "conv_hidden_dims": [512, 512, 512, 512, 512],
                "conv_deep_image_size": 2,
                "conv_norm": True,
                "conv_norm_type": "layer",
                "conv_start_with_mlp": True,
                "conv_activation": None,
                "sdf_latent_size": 128,
                "sdf_hidden_dims": [512, 512, 512],
                "sdf_weight_norm": True,
                "sdf_final_activation": "tanh",
                "sdf_activation": "relu",
                "padding": 0.1,
            },
            "mlp_params": {
                "dims": [512, 512, 512, 512, 512, 512, 512, 512],
                "dropout": None,
                "dropout_prob": 0.0,
                "latent_in": (),
                "weight_norm": True,
                "activation": "relu",
                "final_activation": "tanh",
                "concat_latent_input": True,
            },
        }

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: " f"{list_supported_models()}"
        )
