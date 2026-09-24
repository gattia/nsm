"""
``NSM.models.loader``: templates, ``load_model``, and the architecture keys a config has to
state because no default can be right for them.
"""

import ast
import inspect
import json
import re

import pytest
import torch

import NSM.models.loader as loader
from NSM.models import (
    Decoder,
    ImplicitDecoder,
    TriplanarDecoder,
    get_model_config_template,
    list_supported_models,
    load_model,
)
from NSM.models.triplanar import VAEDecoder

EXPECTED_CLASS = {"triplanar": TriplanarDecoder, "deepsdf": Decoder, "implicit": ImplicitDecoder}
EXTRACTORS = {
    "triplanar": loader._get_triplanar_params,
    "deepsdf": loader._get_deepsdf_params,
    "implicit": loader._get_implicit_params,
}

#: Small enough to build each model type in well under a second.
SMALL = {
    "triplanar": dict(
        latent_size=16, conv_hidden_dims=[8, 8], sdf_hidden_dims=[8], sdf_latent_size=8
    ),
    "deepsdf": dict(latent_size=16, layer_dimensions=[16, 16]),
    "implicit": dict(latent_dim=16, hidden_dim=16, num_layers=3),
}


def small_config(model_type):
    config = get_model_config_template(model_type)
    config.update(SMALL[model_type])
    return config


def test_unknown_model_types_and_missing_inputs_are_refused(tmp_path):
    """
    Fails if ``get_model_config_template`` or ``load_model`` accepts an unknown model type,
    ``list_supported_models`` changes its three types, or ``load_model`` raises anything but
    ``FileNotFoundError`` for a missing checkpoint and ``KeyError`` for an empty config.
    """
    assert set(list_supported_models()) == set(EXPECTED_CLASS)
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model_config_template("invalid_model_type")

    path = str(tmp_path / "empty.pt")
    torch.save({"model": {}}, path)
    config = get_model_config_template("triplanar")
    with pytest.raises(ValueError, match="Unknown model type"):
        load_model(config, path, model_type="invalid_type")
    with pytest.raises(FileNotFoundError):
        load_model(config, str(tmp_path / "missing.pt"), model_type="triplanar")
    with pytest.raises(KeyError, match="Missing required configuration keys"):
        load_model({}, path, model_type="triplanar")


@pytest.mark.parametrize("model_type", sorted(EXPECTED_CLASS))
def test_every_model_type_loads_from_its_template(model_type, tmp_path):
    """
    Fails if ``load_model`` builds the wrong class for a ``model_type``, returns it in
    training mode, or loads a checkpoint that computes differently from the model saved.

    The saved model is built through the same translator ``load_model`` uses, so a
    translator bug shows on both sides and cannot fail this test. The bitwise round trip
    for triplanar is in ``regression/test_model_roundtrip.py``.
    """
    config = small_config(model_type)
    model_class, params = EXTRACTORS[model_type](config)
    assert model_class is EXPECTED_CLASS[model_type]
    torch.manual_seed(0)
    original = model_class(**params).eval()
    path = str(tmp_path / "model.pt")
    torch.save({"model": original.state_dict()}, path)

    loaded = load_model(config, path, model_type=model_type, device="cpu")
    assert type(loaded) is model_class and not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"

    width = config.get("latent_size", config.get("latent_dim"))
    query = torch.randn(10, width + 3)
    with torch.no_grad():
        assert torch.equal(loaded(query), original(query))


class TestConvNormTypeMustBeStated:
    """
    ``conv_norm_type`` and ``conv_activation`` decide what gets built. ``"layer"`` is the
    trained value, and the only thing that makes the VAE nonlinear (ARCHITECTURE §7.1). The
    loader keeps no default for either key, and the triplanar template states the trained
    values.
    """

    def test_a_config_without_either_key_is_refused_and_the_template_states_both(self):
        """
        Fails if the triplanar template stops stating ``conv_norm_type="layer"`` and
        ``conv_activation=None``, or ``load_model`` accepts a triplanar config missing either.
        """
        template = get_model_config_template("triplanar")
        assert (template["conv_norm_type"], template["conv_activation"]) == ("layer", None)
        for key in ("conv_norm_type", "conv_activation"):
            stripped = {k: v for k, v in template.items() if k != key}
            with pytest.raises(KeyError, match=key):
                load_model(stripped, "/nonexistent.pt", model_type="triplanar")

    def test_the_loader_keeps_no_silent_default_for_them(self):
        """
        Fails if ``loader.py`` gains a ``config.get`` with a default for ``conv_norm_type`` or
        ``conv_activation``.

        An AST scan of the module source, so it also sees a default in a branch no other
        test builds.
        """
        defaulted = [
            node.lineno
            for node in ast.walk(ast.parse(inspect.getsource(loader)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("conv_norm_type", "conv_activation")
        ]
        assert defaulted == []

    def test_direct_construction_gets_the_trained_normalization(self):
        """
        Fails if ``TriplanarDecoder`` or ``VAEDecoder`` built without a norm type gets
        BatchNorm instead of the trained LayerNorm.

        The state-dict half shows why a wrong norm fails at load rather than silently:
        ``"batch"`` has every ``"layer"`` key plus BatchNorm's running statistics.
        """

        def norms(model):
            return sorted({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__})

        tiny = dict(latent_dim=8, conv_hidden_dims=[8, 8], sdf_hidden_dims=[8], sdf_latent_size=8)
        assert norms(TriplanarDecoder(**tiny)) == ["LayerNorm"]
        assert norms(VAEDecoder(latent_dim=8, out_features=24, hidden_dims=[8, 8])) == ["LayerNorm"]

        batch = set(TriplanarDecoder(conv_norm_type="batch", **tiny).state_dict())
        layer = set(TriplanarDecoder(conv_norm_type="layer", **tiny).state_dict())
        assert layer < batch
        assert {k.split(".")[-1] for k in batch - layer} <= {
            "running_mean",
            "running_var",
            "num_batches_tracked",
        }


class TestRepairingAnOldTriplanarConfig:
    """
    A config written before Aug 2026 lacks ``padding``, ``conv_activation`` and
    ``conv_norm_type`` (#26, #45). Both shipped production models lack the first two. None
    can be defaulted. One refusal names all three, with a JSON block that repairs the config.
    """

    HISTORICAL = {"padding": 0.1, "conv_activation": None, "conv_norm_type": "layer"}

    def old_config(self):
        return dict(SMALL["triplanar"])

    def test_one_refusal_carries_a_printable_repair(self):
        """
        Fails if ``_get_triplanar_params`` refuses an old config one key at a time, prints
        the refusal with ``\\n`` escapes, or offers a JSON block that does not repair it.

        ``KeyError.__str__`` is ``repr(args[0])``, which prints ``\\n`` escapes and turns the
        JSON block into one unusable line. ``MissingArchitectureKeys`` overrides it and is
        still a ``KeyError``.
        """
        config = self.old_config()
        with pytest.raises(KeyError) as excinfo:
            loader._get_triplanar_params(config)
        printed = str(excinfo.value)
        assert all(key in printed for key in self.HISTORICAL)
        assert "\\n" not in printed and printed.count("\n") > 4

        repair = json.loads(re.search(r"\{.*\}", printed, re.DOTALL).group(0))
        assert repair == self.HISTORICAL
        config.update(repair)
        loader._get_triplanar_params(config)

    def test_direct_construction_gets_the_historical_values(self):
        """
        Fails if ``TriplanarDecoder`` built without the three keys computes differently from
        one built with ``HISTORICAL``. Both shipped models' configs lack ``padding`` and
        ``conv_activation``, so building either directly from its config uses these defaults.

        Compared on a forward pass, because ``padding`` is not a parameter: a wrong default
        still loads a checkpoint strictly, then samples the planes at the wrong scale.
        ``test_shipped_checkpoints`` checks the real models, outside CI.
        """
        tiny = dict(latent_dim=16, conv_hidden_dims=[8, 8], sdf_hidden_dims=[8], sdf_latent_size=8)
        torch.manual_seed(0)
        bare = TriplanarDecoder(**tiny).eval()
        torch.manual_seed(0)
        historical = TriplanarDecoder(**tiny, **self.HISTORICAL).eval()

        torch.manual_seed(1)
        query = torch.cat([torch.randn(1, 16).repeat(64, 1), torch.rand(64, 3) * 2 - 1], dim=1)
        with torch.no_grad():
            assert torch.equal(bare(query), historical(query))
