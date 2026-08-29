from .deep_sdf import *  # noqa: F401,F403  # re-export; see docs/ARCHITECTURE.md star-import trap
from .loader import get_model_config_template, list_supported_models, load_model
from .modulated_periodic_activations import ImplicitDecoder, LinearBlockFactory, SirenBlockFactory
from .triplanar import TriplanarDecoder
from .two_stage import TwoStageDecoder

# What this package exports. Every name below is defined in this package -- the rule is
# mechanical, so nothing is here by opinion. What it leaves out is the accident:
# `from .main import *` and friends also bind `torch`, `os`, `wandb` and whatever else the
# source imported, and a star-import took all of it. See docs/SCOPE.md section 3.3 for why
# this is per-subpackage and not in NSM/__init__.py.
#
# It does NOT unbind those names -- `NSM.datasets.torch` still resolves -- and it is not
# yet the stability tiering of SCOPE section 3.2, which is a separate ruling.
__all__ = [  # noqa: F405 - the deep_sdf names come through the star re-export above
    "Decoder",
    "ImplicitDecoder",
    "LinearBlockFactory",
    "Sine",
    "SirenBlockFactory",
    "TriplanarDecoder",
    "TwoStageDecoder",
    "deep_sdf",
    "get_activation",
    "get_model_config_template",
    "init_weights",
    "list_supported_models",
    "load_model",
    "loader",
    "modulated_periodic_activations",
    "triplanar",
    "two_stage",
]
