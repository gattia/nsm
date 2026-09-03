"""
Every documented option of every model type: does it build, and does it survive one
forward?

``models/`` is the only package in NSM with no test that constructs anything but the
default. The audit's "constructible-but-uncallable configuration" class (five instances,
`ARCHITECTURE.md` section 7) is what that gap hides: a config key is accepted, the model
builds, and the failure arrives on the first forward pass -- or, worse, does not arrive at
all and the model trains degraded.

The matrix below is issue #46's closure criterion written out: **each documented option
value either works or refuses at construction.** Anything else is a defect, and the
strict xfails name which one. The option values come from
``loader.get_model_config_template``, which is what the README tells a reader to start
from, plus the values ``NSM/configs/default_config.json`` actually ships.
"""

import pytest
import torch

from NSM.models.loader import (
    _get_deepsdf_params,
    _get_triplanar_params,
    _get_two_stage_params,
    get_model_config_template,
)

#: Small enough that the whole matrix runs in about a second, large enough that the
#: DeepSDF branch has the eight hidden layers ``PROGRESSIVE_PARAMS`` indexes (5, 6, 7).
LATENT = 8
N_POINTS = 6
N_LAYERS = 8

EXTRACTORS = {
    "deepsdf": _get_deepsdf_params,
    "triplanar": _get_triplanar_params,
    "two_stage": _get_two_stage_params,
}


def small_config(model_type, **overrides):
    """``get_model_config_template`` shrunk to CPU-test size, then overridden."""
    config = get_model_config_template(model_type)
    if model_type == "deepsdf":
        config.update(latent_size=LATENT, layer_dimensions=[16] * N_LAYERS)
    elif model_type == "triplanar":
        config.update(
            latent_size=LATENT * 4,
            conv_hidden_dims=[8, 8],
            sdf_latent_size=12,
            sdf_hidden_dims=[16, 16],
        )
    elif model_type == "two_stage":
        config.update(latent_size=LATENT * 2)
        config["triplanar_params"].update(
            conv_hidden_dims=[8, 8], sdf_latent_size=12, sdf_hidden_dims=[16, 16]
        )
        config["mlp_params"].update(dims=[16] * 4)
    config.update(overrides)
    return config


def build(model_type, **overrides):
    """Construct exactly the way ``load_model`` does, minus the checkpoint."""
    model_class, params = EXTRACTORS[model_type](small_config(model_type, **overrides))
    torch.manual_seed(0)
    return model_class(**params)


def latent_width(model_type, model):
    if model_type == "triplanar":
        return model.latent_dim
    if model_type == "two_stage":
        return model.latent_size
    return model.dims[0] - 3


def forward_once(model_type, model, epoch=None):
    """One forward on the legacy concatenated ``[latent | xyz]`` input every type takes."""
    torch.manual_seed(0)
    width = latent_width(model_type, model)
    latent = torch.randn(1, width).repeat(N_POINTS, 1) * 0.05
    xyz = torch.rand(N_POINTS, 3) * 2 - 1
    model.eval()
    with torch.no_grad():
        return model(torch.cat([latent, xyz], dim=1), epoch=epoch)


def build_and_forward(model_type, epoch=None, **overrides):
    return forward_once(model_type, build(model_type, **overrides), epoch=epoch)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

DEFECT_46 = "https://github.com/gattia/nsm/issues/46"


def broken(reason):
    return pytest.mark.xfail(strict=True, reason=f"#46: {reason} ({DEFECT_46})")


ACTIVATIONS = ["relu", "leaky_relu", "sigmoid", "tanh", "softplus", "elu", "selu", "swish", "sin"]


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_every_hidden_activation_forwards(activation):
    """``get_activation``'s own branch list, minus ``'linear'`` which is its own case."""
    assert build_and_forward("deepsdf", activation=activation).shape == (N_POINTS, 1)


@pytest.mark.parametrize("final_activation", ACTIVATIONS + ["linear"])
def test_every_final_activation_forwards(final_activation):
    """``'linear'`` belongs here and works: ``forward`` guards ``final_activation is None``."""
    assert build_and_forward("deepsdf", final_activation=final_activation).shape == (N_POINTS, 1)


def test_a_linear_hidden_activation_works_or_refuses_at_construction():
    """
    ``get_activation('linear')`` returns ``None``, correctly for the final position and
    fatally for the hidden one, where until Aug 2026 ``forward`` called it (#46).

    Written as "refuses OR forwards" rather than ``pytest.raises`` on purpose: an
    implementation that made hidden ``'linear'`` mean ``nn.Identity`` would also be a fix,
    and this test should not forbid it.
    """
    try:
        model = build("deepsdf", activation="linear")
    except (ValueError, TypeError):
        return
    assert forward_once("deepsdf", model).shape == (N_POINTS, 1)


@pytest.mark.parametrize("epoch", [0, 100, 300, 700, 1300])
def test_progressive_add_depth_forwards_at_every_epoch(epoch):
    """
    ``PROGRESSIVE_PARAMS`` phases layers 5, 6 and 7 in at epochs 200, 600 and 1010, so
    every epoch below 1010 has at least one not-yet-started block. Until Aug 2026
    ``forward_branch_`` returned ``None`` for those and the next layer was handed it, so
    1300 was the only value in this list that worked (#46).
    """
    assert build_and_forward("deepsdf", progressive_add_depth=True, epoch=epoch).shape == (
        N_POINTS,
        1,
    )


def test_progressive_add_depth_refuses_a_forward_with_no_epoch():
    """``self.epoch`` starts as ``None``, and ``None >= int`` is a TypeError from inside."""
    model = build("deepsdf", progressive_add_depth=True)
    with pytest.raises(ValueError, match="epoch"):
        forward_once("deepsdf", model)


def test_a_block_phases_in_continuously_across_its_start_epoch():
    """
    The phase-in weight is ``((epoch - start) / warmup) ** 2``, so at ``epoch == start`` it
    is zero and the block is an identity -- the same thing skipping it does one epoch
    earlier. Until Aug 2026 the warmup branch tested ``start < epoch``, so ``epoch ==
    start`` fell through to applying the block at FULL weight for exactly one epoch before
    dropping back to ``(1/warmup)**2``.

    Asserted as a ratio against the step the phase-in actually takes, not a fixed
    tolerance: the step from ``start`` to ``start + 1`` is what the schedule intends to
    move, and the step across ``start`` must not be larger than it.
    """
    from NSM.models.deep_sdf import PROGRESSIVE_PARAMS

    start = PROGRESSIVE_PARAMS["layers"][5]["start_epoch"]
    model = build("deepsdf", progressive_add_depth=True)
    before, at, after = (
        forward_once("deepsdf", model, epoch=e) for e in (start - 1, start, start + 1)
    )

    across = (at - before).abs().max().item()
    intended = (after - at).abs().max().item()
    assert (
        across <= intended
    ), f"jump across start_epoch {across:.3e} > one warmup step {intended:.3e}"


#: Config keys whose ``Decoder`` argument was deleted in Aug 2026 (#20), with a truthy
#: value and the falsy value every NSM-owned config carries. Both were documented, stored
#: or dropped, and never read by ``forward``: no run has ever had either.
#:
#: ``layers_with_norm`` is deliberately NOT here -- it was not always inert, so it gets
#: ``TestNormLayersWasReachableOnlyOneWay`` below rather than this blanket treatment.
DELETED_OPTIONS = [
    ("xyz_in_all", True, False),
    ("latent_noise_sigma", 0.01, None),
]


@pytest.mark.parametrize(
    "key, truthy, _falsy", DELETED_OPTIONS, ids=[o[0] for o in DELETED_OPTIONS]
)
def test_a_deleted_option_asked_for_is_refused(key, truthy, _falsy):
    """
    ``Decoder`` keeps ``**kwargs``, so deleting the named parameter would put each of these
    straight back to being silently ignored -- the exact defect being fixed. They are
    refused instead, and the message names the config spelling rather than the argument.
    """
    with pytest.raises(TypeError, match=key):
        build("deepsdf", **{key: truthy})


@pytest.mark.parametrize(
    "key, _truthy, falsy", DELETED_OPTIONS, ids=[o[0] for o in DELETED_OPTIONS]
)
def test_a_deleted_option_left_at_its_old_default_is_accepted(key, _truthy, falsy):
    """
    Every NSM-owned config carries these falsy, and a falsy value asked for nothing and got
    nothing -- so refusing it would break configs over a key that never did anything.
    """
    assert build_and_forward("deepsdf", **{key: falsy}).shape == (N_POINTS, 1)


class TestNormLayersWasReachableOnlyOneWay:
    """
    ``norm_layers`` was deleted in Aug 2026 (#46), but it was **not** simply inert, so the
    two cases have to be answered differently -- and getting that wrong breaks real configs.

    The branch that built the LayerNorms is an ``elif`` under ``weight_norm``
    (``deep_sdf.py``, commit ``01d774a``, Jun 2023, whose message says the goal was to
    "separate wieght norm and batch norm so can use both" -- which the ``elif`` is exactly
    what prevents). So with weight norm **on**, the shipped setting, nothing was ever
    appended to the norm list and the key was provably a no-op; with it **off**, LayerNorm
    really was applied, the checkpoint carries ``bn.*`` keys, and a set not starting at
    layer 0 raised ``IndexError`` on the first forward.
    """

    @pytest.mark.parametrize("layers_with_norm", [(0, 1), (1, 2), tuple(range(8))])
    def test_it_is_accepted_where_it_never_did_anything(self, layers_with_norm):
        """
        The case that matters in practice: ``weight_norm: true`` with a full
        ``layers_with_norm``, which is what ``default_config.json`` shipped and what real
        training configs carry. Refusing it would break a config the defect never touched.
        """
        forwarded = build_and_forward(
            "deepsdf", layers_with_norm=layers_with_norm, weight_norm=True
        )
        assert forwarded.shape == (N_POINTS, 1)

    @pytest.mark.parametrize("layers_with_norm", [(0, 1), (1, 2)])
    def test_it_is_refused_where_it_built_layers_the_checkpoint_still_carries(
        self, layers_with_norm
    ):
        """Weight norm off is the configuration whose architecture can no longer be built."""
        with pytest.raises(TypeError, match="bn"):
            build("deepsdf", layers_with_norm=layers_with_norm, weight_norm=False)

    @pytest.mark.parametrize("weight_norm", [True, False])
    def test_an_empty_norm_layers_is_silent_either_way(self, weight_norm):
        forwarded = build_and_forward("deepsdf", layers_with_norm=(), weight_norm=weight_norm)
        assert forwarded.shape == (N_POINTS, 1)


@pytest.mark.parametrize("layer_split", [None, 2])
@pytest.mark.parametrize("objects_per_decoder", [1, 2])
def test_layer_split_forwards(layer_split, objects_per_decoder):
    forwarded = build_and_forward(
        "deepsdf", layer_split=layer_split, objects_per_decoder=objects_per_decoder
    )
    assert forwarded.shape == (N_POINTS, objects_per_decoder)


def test_layer_split_false_is_the_same_model_as_no_layer_split():
    """
    ``default_config.json`` ships ``"layer_split": false``. ``Decoder`` tests
    ``self.layer_split is not None``, and ``False is not None``, so until Aug 2026 every
    layer was split -- which moves every state-dict key from ``layers.N.weight`` to
    ``layers.N.0.weight``, and with ``objects_per_decoder > 1`` builds a different
    architecture entirely (#46). ``False == 0`` in Python, so a value check cannot tell the
    shipped "off" from a deliberate split at layer 0; only ``is`` can.
    """
    absent = build("deepsdf", layer_split=None)
    shipped = build("deepsdf", layer_split=False)
    assert list(shipped.state_dict()) == list(absent.state_dict())


def test_layer_split_zero_still_splits_at_layer_zero():
    """The other half: normalizing ``False`` must not take ``0`` with it."""
    split = build("deepsdf", layer_split=0)
    assert all(
        key.startswith("layers.") and key.split(".")[2].isdigit() for key in split.state_dict()
    )


@pytest.mark.parametrize("concat_latent_input", [False, True])
@pytest.mark.parametrize("layer_latent_in", [(), (4,)])
def test_latent_reinjection_forwards(concat_latent_input, layer_latent_in):
    forwarded = build_and_forward(
        "deepsdf", concat_latent_input=concat_latent_input, layer_latent_in=layer_latent_in
    )
    assert forwarded.shape == (N_POINTS, 1)


@pytest.mark.parametrize("dropout_prob", [0.0, 0.2])
def test_dropout_forwards(dropout_prob):
    forwarded = build_and_forward(
        "deepsdf", layers_with_dropout=list(range(N_LAYERS)), dropout_prob=dropout_prob
    )
    assert forwarded.shape == (N_POINTS, 1)


# --- triplanar ---------------------------------------------------------------


@pytest.mark.parametrize(
    "conv_norm, conv_norm_type", [(False, "batch"), (True, "batch"), (True, "layer")]
)
@pytest.mark.parametrize("conv_start_with_mlp", [False, True])
def test_triplanar_vae_options_forward(conv_norm, conv_norm_type, conv_start_with_mlp):
    forwarded = build_and_forward(
        "triplanar",
        conv_norm=conv_norm,
        conv_norm_type=conv_norm_type,
        conv_start_with_mlp=conv_start_with_mlp,
    )
    assert forwarded.shape == (N_POINTS, 1)


@pytest.mark.parametrize(
    "sum_conv_output_features, conv_pred_sdf",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_triplanar_feature_combination_works_or_refuses(sum_conv_output_features, conv_pred_sdf):
    """
    Four combinations; only the two that sum are correct today.

    Concatenation with ``conv_pred_sdf`` is the one that refuses, and it has no defined
    repair: with three planes concatenated there are three low-frequency SDF channels, one
    per plane, and nothing has ever said how they combine. Until Aug 2026 it built and
    then handed the SDF decoder 17 features where 15 were sized (#45).
    """
    try:
        model = build(
            "triplanar",
            sum_conv_output_features=sum_conv_output_features,
            conv_pred_sdf=conv_pred_sdf,
        )
    except (ValueError, TypeError):
        assert not sum_conv_output_features, "summing must not refuse"
        return
    assert forward_once("triplanar", model).shape == (N_POINTS, 1)


def test_concatenation_uses_all_three_planes():
    """
    ``__init__`` sizes the VAE output by ``sdf_latent_size`` when not summing, and until
    Aug 2026 ``forward_with_plane_features`` sliced ``sdf_latent_size`` **per plane**: xz
    took everything and yz and xy took zero-width slices, so the concatenated result was
    ``torch.equal`` to sampling the xz plane alone -- exact equality, not an approximation,
    which is what makes it assertable in both directions (#45).

    Each plane's slice is asserted to be non-empty as well, because a width that is merely
    *different* from the old one would satisfy the inequality without fixing anything.
    """
    model = build("triplanar", sum_conv_output_features=False)
    torch.manual_seed(0)
    latent = torch.randn(1, model.latent_dim)
    xyz = torch.rand(N_POINTS, 3) * 2 - 1
    model.eval()
    with torch.no_grad():
        planes = model.vae_decoder(latent)[0]
        combined = model.forward_with_plane_features(planes, xyz)
        per_plane = model.sdf_latent_size // 3
        slices = [planes[i * per_plane : (i + 1) * per_plane] for i in range(3)]
        sampled = [
            model.sample_plane_features(xyz, plane, name)
            for plane, name in zip(slices, ("xz", "yz", "xy"))
        ]
    assert combined.shape == (N_POINTS, model.sdf_latent_size)
    assert all(part.shape[1] == per_plane and per_plane > 0 for part in sampled)
    assert torch.equal(combined, torch.cat(sampled, dim=1))
    assert not torch.equal(combined, sampled[0].repeat(1, 3))


def test_the_concatenating_vae_keeps_the_width_it_always_had():
    """
    What makes #45's fix loadable rather than breaking: the VAE emitted
    ``sdf_latent_size`` channels before the fix and emits ``sdf_latent_size`` after, so
    every parameter shape is unchanged and a pre-fix checkpoint still loads. What changed
    is only how those channels are divided among the planes.
    """
    concat = build("triplanar", sum_conv_output_features=False)
    assert concat.vae_decoder.out_features == concat.sdf_latent_size


@pytest.mark.parametrize("objects_per_decoder", [1, 2, 3])
def test_triplanar_multi_object_forwards(objects_per_decoder):
    forwarded = build_and_forward("triplanar", objects_per_decoder=objects_per_decoder)
    assert forwarded.shape == (N_POINTS, objects_per_decoder)


# --- two_stage ---------------------------------------------------------------


def test_two_stage_builds_from_its_own_defaults():
    """
    ``TwoStageDecoder`` with no arguments at all. Until Aug 2026 this raised for every
    argument list, so the type had never been constructible: ``default_mlp_params["dims"]``
    is a tuple and ``Decoder`` did ``[latent_size + 3] + dims`` (#46).
    """
    from NSM.models.two_stage import TwoStageDecoder

    assert isinstance(TwoStageDecoder(), torch.nn.Module)


def test_two_stage_does_not_mutate_its_module_level_defaults():
    """
    ``__init__`` writes ``latent_dim`` and ``n_objects`` into whichever dicts it was
    handed, and the defaults are module-level. Until Aug 2026 one construction -- even a
    failed one -- changed what every later default construction meant, process-wide (#46).
    """
    from NSM.models import two_stage

    before = dict(two_stage.default_triplanar_params)
    try:
        two_stage.TwoStageDecoder(latent_size=64)
    except Exception:
        pass
    assert two_stage.default_triplanar_params == before


def test_two_stage_forwards_from_the_template():
    assert build_and_forward("two_stage").shape == (N_POINTS, 2)
