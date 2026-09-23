"""
The release-time check on the real shipped checkpoints (275 MB and 260 MB, never in CI):

```bash
NSM_SHIPPED_MODELS=/path/to/NSM_MODELS pytest testing/NSM/regression/test_shipped_checkpoints.py
```

Each subdirectory with a ``model_params_config.json`` and a ``model/*.pth`` is one case.
Both shipped configs omit ``padding`` and ``conv_activation``, so ``load_model`` refuses
them (#26, #45). The refusal must name every missing key in one message; the repaired
config must load strictly; and kneepipeline's own construction, fifteen keys by hand with
neither of those two, must build the identical model.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from NSM.models.loader import REQUIRED_ARCHITECTURE_KEYS, load_model
from NSM.models.triplanar import TriplanarDecoder

#: Point this at a directory of trained model folders to run this module.
SHIPPED_MODELS_ENV = "NSM_SHIPPED_MODELS"


def discover():
    root = os.environ.get(SHIPPED_MODELS_ENV, "")
    if not root or not Path(root).is_dir():
        return []
    found = []
    for config in sorted(Path(root).glob("*/model_params_config.json")):
        checkpoints = sorted(config.parent.glob("model/*.pth"))
        if checkpoints:
            found.append(pytest.param(config, checkpoints[-1], id=config.parent.name))
    return found


CASES = discover()

pytestmark = pytest.mark.skipif(
    not CASES, reason=f"set {SHIPPED_MODELS_ENV} to a directory of trained model folders"
)


def repaired(config_path):
    """The config plus what its refusal message says to add, parsed from the message."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        load_model(config, "/nonexistent", device="cpu")
    except KeyError as exc:
        message = exc.args[0]
        block = message[message.index("{") : message.rindex("}") + 1]
        config.update(json.loads(block))
    except Exception:  # the config was already complete; the path is what failed
        pass
    return config


def consumer_style(config, checkpoint):
    """
    ``kneepipeline/steps/run_nsm.py:93-112``, reproduced: fifteen keys by hand, no
    ``padding`` and no ``conv_activation``.
    """
    model = TriplanarDecoder(
        latent_dim=config["latent_size"],
        n_objects=config["objects_per_decoder"],
        conv_hidden_dims=config["conv_hidden_dims"],
        conv_deep_image_size=config["conv_deep_image_size"],
        conv_norm=config["conv_norm"],
        conv_norm_type=config["conv_norm_type"],
        conv_start_with_mlp=config["conv_start_with_mlp"],
        sdf_latent_size=config["sdf_latent_size"],
        sdf_hidden_dims=config["sdf_hidden_dims"],
        sdf_weight_norm=config["weight_norm"],
        sdf_final_activation=config["final_activation"],
        sdf_activation=config["activation"],
        sdf_dropout_prob=config["dropout_prob"],
        sum_sdf_features=config["sum_conv_output_features"],
        conv_pred_sdf=config["conv_pred_sdf"],
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"], strict=True)
    return model.eval()


@pytest.mark.parametrize("config_path,checkpoint", CASES)
class TestAShippedCheckpoint:
    def test_one_message_names_every_key_the_config_lacks(self, config_path, checkpoint):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        missing = [key for key in REQUIRED_ARCHITECTURE_KEYS if key not in config]
        if not missing:
            pytest.skip("this config states every required key")
        with pytest.raises(KeyError) as excinfo:
            load_model(config, str(checkpoint), device="cpu")
        message = str(excinfo.value)
        assert all(key in message for key in missing)
        block = json.loads(message[message.index("{") : message.rindex("}") + 1])
        assert sorted(block) == sorted(missing)

    def test_it_loads_strictly_through_load_model(self, config_path, checkpoint):
        assert load_model(repaired(config_path), str(checkpoint), device="cpu") is not None

    def test_the_consumer_s_own_construction_is_the_same_model(self, config_path, checkpoint):
        """
        Bitwise, on a forward pass -- module types would miss a wrong ``padding``, which
        is not a parameter and changes only where the feature planes get sampled.
        """
        config = repaired(config_path)
        by_loader = load_model(config, str(checkpoint), device="cpu")
        by_hand = consumer_style(config, checkpoint)

        assert list(by_loader.state_dict()) == list(by_hand.state_dict())
        assert by_loader.padding == by_hand.padding

        torch.manual_seed(0)
        latent = torch.randn(1, config["latent_size"]) * 0.05
        query = torch.cat([latent.repeat(64, 1), torch.rand(64, 3) * 2 - 1], dim=1)
        with torch.no_grad():
            assert torch.equal(by_loader(query), by_hand(query))
