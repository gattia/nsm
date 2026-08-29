"""
Repairing a pre-Aug-2026 config: how many attempts does the refusal cost?

``load_model`` requires three triplanar keys that no config written before Aug 2026
carries -- ``padding``, ``conv_activation`` and ``conv_norm_type`` (§8.0.H, issues #26 and
#45). Each refusal is correct and none of them can be defaulted: all three decide what
gets built or how it is sampled, and a checkpoint cannot contradict any of them.

What is measured here is the *delivery*. The three live in three separate ``if``/``raise``
blocks, so a caller repairs one key, re-runs, and is told about the next one. The same
shape sits 100 lines away in the two_stage branch, which loops over two keys and raises
inside the loop.

This is not a hypothetical old config. **Both shipped production models** --
``647_nsm_femur_v0.0.1`` and ``551_nsm_femur_bone_v0.0.1``, the two ``kneepipeline``
ships -- omit ``padding`` and ``conv_activation``, so ``load_model`` on either takes two
refusals before it loads. ``testing/NSM/regression/test_shipped_checkpoints.py`` runs that
against the real weights when they are present; these tests reproduce it in miniature and
need nothing on disk.

The last class is the other half of §8.0.O: the two_stage branch does not merely default
``padding``, it **drops a value the config states**.
"""

import json
import re

import pytest
import torch

from NSM.models.loader import _get_triplanar_params, _get_two_stage_params
from NSM.models.two_stage import TwoStageDecoder

#: The values that reproduce a model trained before Aug 2026, taken from what each
#: refusal message tells the caller to add.
HISTORICAL = {"padding": 0.1, "conv_activation": None, "conv_norm_type": "layer"}

#: Small enough to build in well under a second.
TINY = {
    "conv_hidden_dims": [8, 8],
    "conv_deep_image_size": 2,
    "sdf_latent_size": 8,
    "sdf_hidden_dims": [8],
}


def old_triplanar_config():
    """A config as it was written before Aug 2026: none of the three keys present."""
    return dict(TINY, latent_size=8)


def old_two_stage_config():
    return dict(TINY, latent_size=16, layer_dimensions=[8, 8])


def repair_loop(extractor, config, limit=6):
    """
    Repair ``config`` using only what each refusal names, and count the attempts.

    This is what a caller holding an old config actually does, and the count is the
    number the slice is trying to move. Returns ``(attempts, keys_named_in_order)``.
    """
    named = []
    for attempt in range(1, limit + 1):
        try:
            extractor(config)
        except KeyError as exc:
            missing = [k for k in HISTORICAL if k in exc.args[0] and k not in config]
            assert missing, f"the message named no missing key: {exc.args[0]!r}"
            named.append(tuple(sorted(missing)))
            config.update({k: HISTORICAL[k] for k in missing})
            continue
        return attempt, named
    raise AssertionError(f"still refusing after {limit} attempts: {named}")


class TestRepairingAnOldTriplanarConfig:
    """
    Three refusals, one key each. The fix is one message naming every missing key.
    """

    def test_the_three_keys_are_what_an_old_config_lacks(self):
        """
        The premise, so that a later default does not quietly make this file vacuous.
        """
        config = old_triplanar_config()
        assert [k for k in HISTORICAL if k not in config] == list(HISTORICAL)

    def test_each_refusal_names_exactly_one_key(self):
        """Today's behaviour, stated so the change to it is visible."""
        _, named = repair_loop(_get_triplanar_params, old_triplanar_config())
        assert named == [("padding",), ("conv_activation",), ("conv_norm_type",)]

    @pytest.mark.xfail(
        strict=True, reason="§8.0.O(a): the refusal names one key per attempt, not all three"
    )
    def test_one_refusal_repairs_the_whole_config(self):
        attempts, named = repair_loop(_get_triplanar_params, old_triplanar_config())
        assert attempts == 2, f"repaired in {attempts} attempts, naming {named}"

    @pytest.mark.xfail(
        strict=True, reason="§8.0.O(a): no paste-ready block in the message to parse"
    )
    def test_the_message_carries_a_json_block_that_repairs_the_config(self):
        """
        The message claims a repair; parsing it and applying it is what stops that claim
        drifting from what the code requires. A message a reader has to retype by hand is
        also a message that can go stale without anything noticing.
        """
        config = old_triplanar_config()
        with pytest.raises(KeyError) as excinfo:
            _get_triplanar_params(config)
        block = re.search(r"\{.*\}", excinfo.value.args[0], re.DOTALL)
        assert block, "no JSON object in the refusal message"
        config.update(json.loads(block.group(0)))
        _get_triplanar_params(config)


class TestRepairingAnOldTwoStageConfig:
    """
    The second instance of the same shape. ``_get_two_stage_params`` needs two of the
    three keys and raises on the first one missing.
    """

    def test_each_refusal_names_exactly_one_key(self):
        _, named = repair_loop(_get_two_stage_params, old_two_stage_config())
        assert named == [("conv_norm_type",), ("conv_activation",)]

    @pytest.mark.xfail(
        strict=True, reason="§8.0.O(a): the two_stage loop raises on the first missing key"
    )
    def test_one_refusal_names_both_keys(self):
        attempts, named = repair_loop(_get_two_stage_params, old_two_stage_config())
        assert attempts == 2, f"repaired in {attempts} attempts, naming {named}"


class TestTwoStageDropsAStatedPadding:
    """
    #26's defect in the branch §8.0.H did not open, and in its worse form.

    #26 was "a ``padding`` the config did not state was silently defaulted". Here the
    config *does* state it and the value never arrives: ``_get_two_stage_params`` builds
    its ``triplanar_params`` without a ``padding`` key at all, so ``TriplanarDecoder``'s
    constructor default wins. ``padding`` scales query coordinates before they index the
    feature planes and is not a learned parameter, so ``load_state_dict(strict=True)``
    cannot contradict it -- the model loads clean and samples at the wrong scale.

    Nothing existing is affected: both shipped models are ``model_type: "triplanar"``,
    the shipped ``default_config.json`` is triplanar with ``padding: 0.1`` stated, and no
    two_stage config exists in this repo or in either consumer. That is what makes
    refusing it at the release boundary cheap.
    """

    @staticmethod
    def config(**overrides):
        return dict(
            old_two_stage_config(), conv_norm_type="layer", conv_activation=None, **overrides
        )

    def test_the_triplanar_branch_forwards_a_stated_padding(self):
        """The sibling path, for contrast: this is what two_stage should do."""
        config = dict(old_triplanar_config(), **HISTORICAL)
        config["padding"] = 0.35
        _, params = _get_triplanar_params(config)
        assert params["padding"] == 0.35

    @pytest.mark.xfail(strict=True, reason="§8.0.O: two_stage drops a stated padding (#26)")
    def test_a_stated_padding_reaches_the_decoder(self):
        cls, params = _get_two_stage_params(self.config(padding=0.35))
        assert params["triplanar_params"].get("padding") == 0.35
        assert cls(**params).triplanar.padding == 0.35

    def test_today_it_silently_becomes_the_constructor_default(self):
        """The behaviour as it stands, so the xfail above cannot be the only record."""
        cls, params = _get_two_stage_params(self.config(padding=0.35))
        assert "padding" not in params["triplanar_params"]
        assert cls(**params).triplanar.padding == 0.1

    @pytest.mark.xfail(
        strict=True, reason="§8.0.O: two_stage does not require padding as triplanar does"
    )
    def test_a_two_stage_config_without_padding_is_refused(self):
        with pytest.raises(KeyError, match="padding"):
            _get_two_stage_params(self.config())


class TestNormTypeConstructorDefault:
    """
    §8.0.O(b): the constructor still defaults ``conv_norm_type`` to ``"batch"`` while
    every config that has ever trained says ``"layer"``.

    Unreachable through ``load_model`` since §8.0.H made the key required -- and reachable
    by direct construction, which is exactly what the downstream consumer does
    (``kneepipeline/steps/run_nsm.py`` builds ``TriplanarDecoder(**params)`` by hand).
    That consumer passes ``conv_norm_type`` explicitly, so the change costs it nothing;
    what the default silently cost was a fresh model built without one.

    Changing a public-stable class's signature is a Breaking change, which is why this
    waited for the release rather than riding with §8.0.H.
    """

    @staticmethod
    def norm_types(model):
        return sorted({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__})

    def test_the_config_path_cannot_reach_any_default(self):
        """True before and after: the refusal is what makes changing the default safe."""
        with pytest.raises(KeyError, match="conv_norm_type"):
            _get_triplanar_params(dict(old_triplanar_config(), padding=0.1, conv_activation=None))

    @pytest.mark.xfail(strict=True, reason='§8.0.O(b): the constructor still defaults to "batch"')
    def test_direct_construction_gets_the_trained_normalization(self):
        from NSM.models.triplanar import TriplanarDecoder

        assert self.norm_types(TriplanarDecoder(latent_dim=8, **TINY)) == ["LayerNorm"]

    @pytest.mark.xfail(strict=True, reason='§8.0.O(b): VAEDecoder still defaults to "batch"')
    def test_the_vae_default_matches(self):
        from NSM.models.triplanar import VAEDecoder

        assert self.norm_types(VAEDecoder(latent_dim=8, out_features=24, hidden_dims=[8, 8])) == [
            "LayerNorm"
        ]

    def test_the_two_normalizations_are_not_interchangeable_on_disk(self):
        """
        Why the default matters and why a mismatch is nonetheless loud: BatchNorm2d
        carries ``running_mean``/``running_var``/``num_batches_tracked`` that LayerNorm
        does not, so a strict load reports missing or unexpected keys in either
        direction. The silent cost is a freshly built model, not a reloaded one.
        """
        from NSM.models.triplanar import TriplanarDecoder

        batch = set(TriplanarDecoder(latent_dim=8, conv_norm_type="batch", **TINY).state_dict())
        layer = set(TriplanarDecoder(latent_dim=8, conv_norm_type="layer", **TINY).state_dict())
        assert layer < batch
        assert all(
            k.split(".")[-1] in {"running_mean", "running_var", "num_batches_tracked"}
            for k in batch - layer
        )

    def test_a_consumer_style_build_is_unaffected_by_the_default(self):
        """
        The consumer's own construction, reproduced: ``kneepipeline`` reads
        ``conv_norm_type`` out of the model config and passes it, so its model is
        identical whatever the signature says. Asserted on the forward output, not just
        the module types, because that is what a wrong norm would actually change.
        """
        from NSM.models.triplanar import TriplanarDecoder

        def consumer_model():
            torch.manual_seed(0)
            return TriplanarDecoder(latent_dim=8, conv_norm_type="layer", **TINY).eval()

        torch.manual_seed(1)
        query = torch.cat([torch.randn(4, 8) * 0.05, torch.rand(4, 3) * 2 - 1], dim=1)
        with torch.no_grad():
            assert torch.equal(consumer_model()(query), consumer_model()(query))
