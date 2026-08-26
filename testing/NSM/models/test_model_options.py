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
    _get_implicit_params,
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
    "implicit": _get_implicit_params,
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
    elif model_type == "implicit":
        config.update(latent_dim=LATENT, hidden_dim=16, num_layers=4)
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
    if model_type == "implicit":
        return model.latent_dim
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


@broken("activation='linear' returns a bare None that forward then calls")
def test_a_linear_hidden_activation_works_or_refuses_at_construction():
    """
    ``get_activation('linear')`` returns ``None``, correctly for the final position and
    fatally for the hidden one. Either it refuses while building or it forwards; today it
    does neither.
    """
    try:
        model = build("deepsdf", activation="linear")
    except (ValueError, TypeError):
        return
    assert forward_once("deepsdf", model).shape == (N_POINTS, 1)


@pytest.mark.parametrize("epoch", [0, 100, 300, 700, 1300])
def test_progressive_add_depth_forwards_at_every_epoch(epoch):
    """
    ``PROGRESSIVE_PARAMS`` phases layers 5, 6 and 7 in at epochs 200, 600 and 1010, so an
    epoch below 1010 has at least one not-yet-started block. ``forward_branch_`` returns
    ``None`` for one of those, and the next layer is handed it.

    1300 (past every ``start_epoch``) is the only value that works today, which is why it
    is not xfailed: it is what proves the rest is a start-condition defect and not the
    option being broken outright.
    """
    if epoch < 1010:
        pytest.xfail(f"#46: progressive_add_depth propagates None below epoch 1010 ({DEFECT_46})")
    assert build_and_forward("deepsdf", progressive_add_depth=True, epoch=epoch).shape == (
        N_POINTS,
        1,
    )


@pytest.mark.parametrize(
    "layers_with_norm, weight_norm",
    [
        ((0, 1), False),
        ((0, 1), True),
        pytest.param(
            (1, 2),
            False,
            marks=broken("norm_layers indexes self.bn by absolute layer index"),
        ),
        ((1, 2), True),
    ],
)
def test_norm_layers_work_or_refuse(layers_with_norm, weight_norm):
    """
    ``self.bn`` is appended to once per norm layer and read as ``self.bn[layer_idx]``, so
    any set not starting at layer 0 indexes past the end -- but only with weight-norm off,
    because with it on nothing is ever appended and the option is silently inert instead.
    Both halves are pinned: the shipped configuration (weight_norm on) is the inert one.
    """
    forwarded = build_and_forward(
        "deepsdf", layers_with_norm=layers_with_norm, weight_norm=weight_norm
    )
    assert forwarded.shape == (N_POINTS, 1)


@pytest.mark.parametrize("layer_split", [None, 2])
@pytest.mark.parametrize("objects_per_decoder", [1, 2])
def test_layer_split_forwards(layer_split, objects_per_decoder):
    forwarded = build_and_forward(
        "deepsdf", layer_split=layer_split, objects_per_decoder=objects_per_decoder
    )
    assert forwarded.shape == (N_POINTS, objects_per_decoder)


@pytest.mark.xfail(
    strict=True,
    reason="#46: layer_split=False is `is not None`, so it means split-at-layer-0",
)
def test_layer_split_false_is_the_same_model_as_no_layer_split():
    """
    ``default_config.json`` ships ``"layer_split": false``. ``Decoder`` tests
    ``self.layer_split is not None``, and ``False is not None``, so every layer is split --
    which moves every state-dict key from ``layers.N.weight`` to ``layers.N.0.weight``.
    ``False == 0`` in Python, so a value check cannot tell the shipped "off" from a
    deliberate split at layer 0; only ``is`` can.
    """
    absent = build("deepsdf", layer_split=None)
    shipped = build("deepsdf", layer_split=False)
    assert list(shipped.state_dict()) == list(absent.state_dict())


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
        pytest.param(
            False,
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="#45: concatenation + conv_pred_sdf builds and forwards into a shape error",
            ),
        ),
    ],
)
def test_triplanar_feature_combination_works_or_refuses(sum_conv_output_features, conv_pred_sdf):
    """
    Four combinations; only the two that sum are correct today.

    Concatenation with ``conv_pred_sdf`` is broken past the slicing #45 describes and has
    no defined repair: with three planes concatenated there are three low-frequency SDF
    channels, one per plane, and nothing has ever said how they combine. It builds, then
    hands the SDF decoder 17 features where 15 were sized. Either it refuses at
    construction or it forwards.
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


@pytest.mark.xfail(
    strict=True,
    reason="#45: sum_sdf_features=False slices sdf_latent_size per plane, so yz and xy get 0 channels",
)
def test_concatenation_must_not_reduce_to_one_plane():
    """
    ``__init__`` sizes the VAE output by ``sdf_latent_size`` when not summing;
    ``forward_with_plane_features`` then slices ``sdf_latent_size`` **per plane**, so xz
    takes everything and yz and xy take zero-width slices. The concatenated result is
    ``torch.equal`` to sampling the xz plane alone -- exact equality, not an approximation,
    which is what makes this assertable.
    """
    model = build("triplanar", sum_conv_output_features=False)
    torch.manual_seed(0)
    latent = torch.randn(1, model.latent_dim)
    xyz = torch.rand(N_POINTS, 3) * 2 - 1
    model.eval()
    with torch.no_grad():
        planes = model.vae_decoder(latent)[0]
        combined = model.forward_with_plane_features(planes, xyz)
        per_plane = model.sdf_latent_size + model.conv_pred_sdf
        xz_alone = model.sample_plane_features(xyz, planes[:per_plane], "xz")
    assert not torch.equal(combined, xz_alone), "yz and xy contributed nothing"


@pytest.mark.parametrize("objects_per_decoder", [1, 2, 3])
def test_triplanar_multi_object_forwards(objects_per_decoder):
    forwarded = build_and_forward("triplanar", objects_per_decoder=objects_per_decoder)
    assert forwarded.shape == (N_POINTS, objects_per_decoder)


# --- two_stage and implicit --------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#46: TwoStageDecoder() concatenates a list and a tuple")
def test_two_stage_builds_from_its_own_defaults():
    """
    ``TwoStageDecoder`` with no arguments at all: ``default_mlp_params["dims"]`` is a
    tuple and ``Decoder`` does ``[latent_size + 3] + dims``. It raises for every argument
    list, so the type has never been constructible.
    """
    from NSM.models.two_stage import TwoStageDecoder

    assert isinstance(TwoStageDecoder(), torch.nn.Module)


@pytest.mark.xfail(
    strict=True, reason="#46: TwoStageDecoder mutates its module-level default dicts"
)
def test_two_stage_does_not_mutate_its_module_level_defaults():
    """
    ``__init__`` writes ``latent_dim`` and ``n_objects`` into whichever dicts it was
    handed, and the defaults are module-level. One construction -- even a failed one --
    changes what every later default construction means, process-wide.
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


@pytest.mark.parametrize("block_type", ["linear", "siren"])
@pytest.mark.parametrize("modulation", [False, True])
def test_implicit_options_forward(block_type, modulation):
    forwarded = build_and_forward("implicit", block_type=block_type, modulation=modulation)
    assert forwarded.shape == (N_POINTS, 1)


@pytest.mark.parametrize("final_activation", ["sigmoid", "tanh", "linear"])
def test_implicit_final_activations_forward(final_activation):
    forwarded = build_and_forward("implicit", final_activation=final_activation)
    assert forwarded.shape == (N_POINTS, 1)
