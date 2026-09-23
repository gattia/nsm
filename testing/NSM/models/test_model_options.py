"""
Every documented option of every model type builds and forwards, or refuses at
construction (#46's closure criterion). The option values come from
``loader.get_model_config_template`` and from ``NSM/configs/default_config.json``.
"""

import inspect

import pytest
import torch

from NSM.models.deep_sdf import DELETED_DECODER_ARGUMENTS, PROGRESSIVE_PARAMS, Decoder
from NSM.models.loader import (
    _get_deepsdf_params,
    _get_implicit_params,
    _get_triplanar_params,
    get_model_config_template,
)
from NSM.models.triplanar import TriplanarDecoder

#: Eight hidden layers, because ``PROGRESSIVE_PARAMS`` phases in layers 5, 6 and 7.
LATENT = 8
N_POINTS = 6
N_LAYERS = 8

EXTRACTORS = {
    "deepsdf": _get_deepsdf_params,
    "triplanar": _get_triplanar_params,
    "implicit": _get_implicit_params,
}


def build(model_type, **overrides):
    """The template shrunk to CPU size, built the way ``load_model`` builds it."""
    config = get_model_config_template(model_type)
    config.update(
        {
            "deepsdf": dict(latent_size=LATENT, layer_dimensions=[16] * N_LAYERS),
            "triplanar": dict(
                latent_size=LATENT * 4,
                conv_hidden_dims=[8, 8],
                sdf_latent_size=12,
                sdf_hidden_dims=[16, 16],
            ),
            "implicit": dict(latent_dim=LATENT, hidden_dim=16, num_layers=4),
        }[model_type]
    )
    config.update(overrides)
    model_class, params = EXTRACTORS[model_type](config)
    torch.manual_seed(0)
    return model_class(**params)


def forward(model, epoch=None):
    """One forward on the concatenated ``[latent | xyz]`` input every type takes."""
    width = model.latent_dim if hasattr(model, "latent_dim") else model.dims[0] - 3
    torch.manual_seed(0)
    latent = torch.randn(1, width).repeat(N_POINTS, 1) * 0.05
    xyz = torch.rand(N_POINTS, 3) * 2 - 1
    model.eval()
    with torch.no_grad():
        return model(torch.cat([latent, xyz], dim=1), epoch=epoch)


ACTIVATIONS = ["relu", "leaky_relu", "sigmoid", "tanh", "softplus", "elu", "selu", "swish", "sin"]


def test_every_deepsdf_option_forwards():
    """
    Fails if ``Decoder`` cannot build or forward, or returns the wrong output width, under a
    documented activation, final activation, ``layer_split`` and object count, latent
    re-injection or dropout option (#46).

    Hidden ``'linear'`` is the one activation that may refuse instead: ``get_activation``
    returns ``None`` for it, which is right for the final layer and cannot run in a hidden
    one. Mapping it to ``nn.Identity`` would also be a fix, so either is allowed. Forward
    runs in eval mode, so dropout is built but never applied.
    """
    options = [dict(activation=a) for a in ACTIVATIONS]
    options += [dict(final_activation=a) for a in ACTIVATIONS + ["linear"]]
    options += [
        dict(layer_split=split, objects_per_decoder=objects)
        for split in (None, 2)
        for objects in (1, 2)
    ]
    options += [
        dict(concat_latent_input=concat, layer_latent_in=latent_in)
        for concat in (False, True)
        for latent_in in ((), (4,))
    ]
    options += [dict(layers_with_dropout=list(range(N_LAYERS)), dropout_prob=p) for p in (0.0, 0.2)]
    for option in options:
        objects = option.get("objects_per_decoder", 1)
        assert forward(build("deepsdf", **option)).shape == (N_POINTS, objects), option

    try:
        model = build("deepsdf", activation="linear")
    except (ValueError, TypeError):
        return
    assert forward(model).shape == (N_POINTS, 1)


class TestProgressiveAddDepth:
    def test_it_forwards_at_every_epoch_and_refuses_none(self):
        """
        Fails if ``Decoder(progressive_add_depth=True)`` cannot forward at an epoch before one
        of its blocks starts, or accepts a forward with no ``epoch`` (#46).

        Every epoch below 1010 has a block that has not started.
        """
        model = build("deepsdf", progressive_add_depth=True)
        for epoch in (0, 100, 300, 700, 1300):
            assert forward(model, epoch=epoch).shape == (N_POINTS, 1)
        with pytest.raises(ValueError, match="epoch"):
            forward(build("deepsdf", progressive_add_depth=True))

    def test_a_block_phases_in_continuously_across_its_start_epoch(self):
        """
        Fails if ``Decoder.progressive_layer`` applies a block at full weight when ``epoch``
        equals its ``start_epoch`` (KNOWN_ISSUES History 14).

        The weight is ``((epoch - start) / warmup) ** 2``, zero at ``start``. The step across
        ``start`` must be no larger than the next warmup step. Measured: 0 across ``start``
        and 3.5e-06 after it. The full-weight bug steps 8.2e-02 both times, so it fails by
        only 3.5e-06.
        """
        start = PROGRESSIVE_PARAMS["layers"][5]["start_epoch"]
        model = build("deepsdf", progressive_add_depth=True)
        before, at, after = (forward(model, epoch=e) for e in (start - 1, start, start + 1))
        assert (at - before).abs().max() <= (after - at).abs().max()


def test_deleted_options_refuse_when_asked_for_and_pass_when_off():
    """
    Fails if the deepsdf path accepts a truthy ``xyz_in_all`` or ``latent_noise_sigma``, or
    ``layers_with_norm`` under ``weight_norm=False``, or refuses the falsy values configs on
    disk carry (#20).

    ``forward`` never read the first two, so no run used them. ``layers_with_norm`` built
    nothing under weight norm (the shipped setting), so it is accepted there. With weight
    norm off it built ``bn.*`` layers a checkpoint carries, an architecture that can no
    longer be built.
    """
    for key, truthy, falsy in (("xyz_in_all", True, False), ("latent_noise_sigma", 0.01, None)):
        with pytest.raises(TypeError, match=key):
            build("deepsdf", **{key: truthy})
        assert forward(build("deepsdf", **{key: falsy})).shape == (N_POINTS, 1)

    for layers in ((0, 1), (1, 2), tuple(range(8)), ()):
        assert forward(build("deepsdf", layers_with_norm=layers, weight_norm=True)) is not None
    assert forward(build("deepsdf", layers_with_norm=(), weight_norm=False)) is not None
    for layers in ((0, 1), (1, 2)):
        with pytest.raises(TypeError, match="bn"):
            build("deepsdf", layers_with_norm=layers, weight_norm=False)


def test_layer_split_false_is_the_same_model_as_no_layer_split():
    """
    Fails if ``Decoder`` treats ``layer_split=False`` as a split at layer 0, which moves every
    state-dict key, or stops splitting at ``layer_split=0`` (KNOWN_ISSUES History 14).

    ``default_config.json`` ships ``"layer_split": false``. ``False == 0``, so only ``is``
    tells the shipped "off" from a deliberate split at layer 0.
    """
    absent = list(build("deepsdf", layer_split=None).state_dict())
    assert list(build("deepsdf", layer_split=False).state_dict()) == absent
    at_zero = build("deepsdf", layer_split=0).state_dict()
    assert all(key.split(".")[2].isdigit() for key in at_zero)


class TestTriplanar:
    def test_every_vae_option_and_object_count_forwards(self):
        """
        Fails if ``TriplanarDecoder`` cannot build or forward for a combination of
        ``conv_norm``, ``conv_norm_type`` and ``conv_start_with_mlp``, or returns the wrong
        width for 2 or 3 objects.
        """
        for norm, norm_type in ((False, "batch"), (True, "batch"), (True, "layer")):
            for start_with_mlp in (False, True):
                model = build(
                    "triplanar",
                    conv_norm=norm,
                    conv_norm_type=norm_type,
                    conv_start_with_mlp=start_with_mlp,
                )
                assert forward(model).shape == (N_POINTS, 1)
        for objects in (2, 3):
            assert forward(build("triplanar", objects_per_decoder=objects)).shape == (
                N_POINTS,
                objects,
            )

    def test_triplanar_feature_combination_works_or_refuses(self):
        """
        Fails if ``TriplanarDecoder`` builds with ``sum_conv_output_features: false`` and
        ``conv_pred_sdf: true``, or cannot forward in the other three combinations
        (KNOWN_ISSUES History 15).

        Concatenating with ``conv_pred_sdf`` gives three SDF channels, one per plane, and
        nothing defines how they combine.
        """
        for summed in (True, False):
            for pred_sdf in (False, True):
                if not summed and pred_sdf:
                    with pytest.raises((ValueError, TypeError)):
                        build("triplanar", sum_conv_output_features=False, conv_pred_sdf=True)
                    continue
                model = build("triplanar", sum_conv_output_features=summed, conv_pred_sdf=pred_sdf)
                assert forward(model).shape == (N_POINTS, 1)

    def test_concatenation_uses_all_three_planes(self):
        """
        Fails if ``TriplanarDecoder.forward_with_plane_features``, when concatenating, does
        not sample xz, yz and xy each from its own ``sdf_latent_size // 3`` channels, or the
        VAE's output width changes (KNOWN_ISSUES History 15).

        The VAE emits ``sdf_latent_size`` channels so pre-#45 checkpoints load.
        """
        model = build("triplanar", sum_conv_output_features=False)
        assert model.vae_decoder.out_features == model.sdf_latent_size

        torch.manual_seed(0)
        latent, xyz = torch.randn(1, model.latent_dim), torch.rand(N_POINTS, 3) * 2 - 1
        with torch.no_grad():
            planes = model.vae_decoder(latent)[0]
            combined = model.forward_with_plane_features(planes, xyz)
            width = model.sdf_latent_size // 3
            sampled = [
                model.sample_plane_features(xyz, planes[i * width : (i + 1) * width], name)
                for i, name in enumerate(("xz", "yz", "xy"))
            ]
        assert width > 0 and all(part.shape[1] == width for part in sampled)
        assert torch.equal(combined, torch.cat(sampled, dim=1))
        assert not torch.equal(combined, sampled[0].repeat(1, 3))


def test_every_implicit_option_forwards():
    """
    Fails if ``ImplicitDecoder`` cannot build or forward with either ``block_type``, with or
    without ``modulation``, or with a ``sigmoid``, ``tanh`` or ``linear`` final activation.

    Forward only, under ``no_grad``, so a failure in backward does not show.
    """
    for block_type in ("linear", "siren"):
        for modulation in (False, True):
            model = build("implicit", block_type=block_type, modulation=modulation)
            assert forward(model).shape == (N_POINTS, 1)
    for final in ("sigmoid", "tanh", "linear"):
        assert forward(build("implicit", final_activation=final)).shape == (N_POINTS, 1)


class TestTheEvidenceForSlicesThatOwnTheFix:
    def test_the_implicit_translator_ignores_the_activation_both_siblings_read(self):
        """
        Fails if ``_get_implicit_params`` starts honouring ``activation`` for the linear
        ``block_type``, or ``_get_deepsdf_params`` stops forwarding it.

        The implicit half pins a known defect: ``LinearBlockFactory()`` is built at its ReLU
        default whatever ``activation`` says. ``SCOPE`` §1 folds any fix into the
        registration work, and that fix turns this test red.
        """
        _, params = _get_implicit_params(
            dict(latent_dim=8, hidden_dim=8, num_layers=2, block_type="linear", activation="sin")
        )
        assert params["block_factory"].activation_cls is torch.nn.ReLU
        _, params = _get_deepsdf_params(
            dict(latent_size=16, layer_dimensions=[8], activation="sin")
        )
        assert params["activation"] == "sin"


def test_the_constructors_refuse_unknown_keywords():
    """
    Fails if ``Decoder`` or ``TriplanarDecoder`` accepts a misspelled keyword, ``Decoder``
    refuses ``None`` for a deleted argument, or ``TriplanarDecoder`` loses one of the 15
    keywords kneepipeline passes it (#26).

    A misspelled ``padding`` that is accepted builds the model at its default and samples
    the planes at the wrong scale. The 15 keys are the constructor call in kneepipeline's
    ``steps/run_nsm.py``.
    """
    with pytest.raises(TypeError, match="paddding"):
        Decoder(latent_size=LATENT, dims=[16, 16], paddding=0.35)
    with pytest.raises(TypeError, match="paddding"):
        TriplanarDecoder(
            latent_dim=LATENT, conv_hidden_dims=[16], sdf_hidden_dims=[16], paddding=0.35
        )
    for deleted in DELETED_DECODER_ARGUMENTS:
        assert Decoder(latent_size=LATENT, dims=[16, 16], **{deleted: None}) is not None

    consumer_keys = set(
        """
        latent_dim n_objects conv_hidden_dims conv_deep_image_size conv_norm conv_norm_type
        conv_start_with_mlp sdf_latent_size sdf_hidden_dims sdf_weight_norm
        sdf_final_activation sdf_activation sdf_dropout_prob sum_sdf_features conv_pred_sdf
        """.split()
    )
    assert consumer_keys <= set(inspect.signature(TriplanarDecoder.__init__).parameters)
