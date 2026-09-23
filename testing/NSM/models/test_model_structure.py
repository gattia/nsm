"""
Structural facts about ``models/`` that no fix may silently change. Two are defects that
cannot be repaired without breaking shipped checkpoints.
"""

import pytest
import torch
from torch import nn

from NSM.models.triplanar import VAEDecoder

LATENT = 16
HIDDEN = [8, 8]


def build_vae(**overrides):
    torch.manual_seed(0)
    kwargs = dict(latent_dim=LATENT, out_features=12, hidden_dims=list(HIDDEN))
    kwargs.update(overrides)
    return VAEDecoder(**kwargs).eval()


def additivity_error(vae, alpha=0.3):
    """
    How far the stack is from affine, and the value scale to read it against. The final
    ``Tanh`` is swapped for ``Identity``: the question is the stack's own nonlinearity.
    """
    final = vae.decoder[-1]
    saved, final[1] = final[1], nn.Identity()
    try:
        torch.manual_seed(1)
        x1, x2 = torch.randn(1, LATENT), torch.randn(1, LATENT)
        with torch.no_grad():
            mixed = vae(alpha * x1 + (1 - alpha) * x2)
            separate = alpha * vae(x1) + (1 - alpha) * vae(x2)
            return (mixed - separate).abs().max().item(), separate.abs().max().item()
    finally:
        final[1] = saved


ACTIVATION_TYPES = (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU, nn.ELU)


def conv_stack_activations(vae):
    """The pointwise activations inside the conv stack, excluding the final ``Tanh``."""
    return [m for m in vae.decoder[:-1].modules() if isinstance(m, ACTIVATION_TYPES)]


class TestTheVAEHasNoActivation:
    """
    ``VAEDecoder`` built an activation and never appended it: a leaked loop variable, in
    every shipped model. The default is still no activation, because that is the
    architecture every checkpoint was fitted as: ``ConvTranspose2d -> norm`` x N, then
    ``Conv2d -> Tanh``. ``docs/ARCHITECTURE.md`` §7.1 has the account.
    """

    def test_the_default_stack_has_only_the_final_tanh(self):
        vae = build_vae()
        assert conv_stack_activations(vae) == [], "the default changed; checkpoints stop loading"
        assert isinstance(vae.decoder[-1][0], nn.Conv2d)
        assert isinstance(vae.decoder[-1][1], nn.Tanh)
        with pytest.raises(TypeError, match="activation"):
            build_vae(activation="relu")

    def test_the_stack_is_affine_except_where_layernorm_saves_it(self):
        """
        ``"batch"`` evaluates affine, so the conv stack collapses to one map. ``"layer"``,
        which both shipped models use, is nonlinear only because LayerNorm divides by a
        standard deviation of its own input.
        """
        for overrides, affine in (
            ({"norm": True, "norm_type": "batch"}, True),
            ({"norm": False}, True),
            ({"norm": True, "norm_type": "layer"}, False),
        ):
            error, scale = additivity_error(build_vae(**overrides))
            assert (error / scale < 1e-6) is affine, (overrides, error / scale)


class TestTheOptInConvActivation:
    """
    ``conv_activation`` is opt-in because ``nn.Sequential`` names children by position:
    inserting an activation renumbers every later key, so a pre-Aug-2026 checkpoint loads
    only at the default.
    """

    def test_the_default_is_the_historical_architecture_and_nothing_else_loads_it(self):
        torch.manual_seed(6)
        historical = build_vae()
        torch.manual_seed(6)
        explicit = build_vae(conv_activation=None)
        x = torch.randn(2, LATENT)
        with torch.no_grad():
            assert torch.equal(historical(x), explicit(x))

        checkpoint = historical.state_dict()
        explicit.load_state_dict(checkpoint, strict=True)
        with pytest.raises(RuntimeError, match="Missing key"):
            build_vae(conv_activation="leaky_relu").load_state_dict(checkpoint, strict=True)

    def test_an_activation_goes_after_each_norm(self):
        """
        ``conv -> norm -> activation`` is provisional (``NSM_TRAINING_IDEAS.md`` Idea 13),
        and is pinned so that changing it is a decision.
        """
        for activation in ("relu", "leaky_relu", "swish", "elu"):
            vae = build_vae(hidden_dims=[8, 8, 8], conv_activation=activation)
            assert len(conv_stack_activations(vae)) == 3
            with torch.no_grad():
                assert vae(torch.randn(2, LATENT)).shape[1] == vae.out_features
        for norm_type, norm in (("layer", "LayerNorm"), ("batch", "BatchNorm2d")):
            vae = build_vae(conv_activation="leaky_relu", norm_type=norm_type)
            assert [type(m).__name__ for m in vae.decoder[:3]] == [
                "ConvTranspose2d",
                norm,
                "LeakyReLU",
            ]

    def test_unknown_and_linear_are_refused(self):
        """``'linear'`` would silently mean the historical stack under a name that reads
        like a choice, so the refusal points at ``None``."""
        with pytest.raises(ValueError, match="Unknown activation"):
            build_vae(conv_activation="not_an_activation")
        with pytest.raises(ValueError, match="None"):
            build_vae(conv_activation="linear")


class TestWhatLayerNormActuallySupplies:
    """
    The shipped models are nonlinear only through LayerNorm, so its kind of nonlinearity
    decides what the missing activation costs. Only its division is nonlinear, a radial
    projection: it cannot zero a feature or form a decision boundary.
    """

    def test_normalization_is_over_the_whole_feature_map_not_per_position(self):
        """
        Over the full ``(C, H, W)``: one scale per sample for the whole map. ConvNeXt's
        per-position convention would give a per-location gain; this is the weaker kind.
        """
        norms = [m for m in build_vae(norm_type="layer").decoder if isinstance(m, nn.LayerNorm)]
        assert norms, "the layer variant stopped building LayerNorms"
        assert all(len(m.normalized_shape) == 3 for m in norms), [m.normalized_shape for m in norms]

    def test_the_latent_magnitude_is_not_discarded(self):
        """
        ``LN(cx) == LN(x)``, so a stack of linear maps into LayerNorm would be blind to
        ``||z||`` and the L2 prior would cost nothing. The biases in ``fc`` and the first
        ``ConvTranspose2d`` break that. Theory for the homogeneous case (an inert norm
        penalty, spherical interpolation) does not apply here.
        """
        vae = build_vae(norm_type="layer")
        assert vae.fc.bias is not None and vae.decoder[0].bias is not None

        torch.manual_seed(4)
        z = torch.randn(1, LATENT)
        with torch.no_grad():
            once, twice = vae(z), vae(2 * z)
        relative = (once - twice).abs().max().item() / once.abs().max().item()
        assert relative > 1e-2, f"the decoder became blind to ||z||: {relative:.2e}"

    def test_the_data_dependence_of_the_gain_attenuates_with_depth(self):
        """
        A per-sample sigma that never moves is a fixed affine map. Across latents spanning
        the fitted production range of norms (``NSM_TRAINING_IDEAS.md`` Idea 4), the spread
        is real at the first LayerNorm and near 1 at the last. On the shipped 647 model:
        1.71x, 1.30x, 1.15x, 1.02x, 1.00x. The ordering is asserted, not the values.
        """
        vae = build_vae(hidden_dims=[16] * 5, norm_type="layer")

        torch.manual_seed(5)
        latents = torch.randn(16, LATENT)
        norms = torch.linspace(4.0, 10.0, 16)[:, None]
        latents = latents / latents.norm(dim=1, keepdim=True) * norms

        seen = {}

        def record(index):
            def hook(module, inputs, output):
                x = inputs[0]
                seen.setdefault(index, []).append(x.std(dim=tuple(range(1, x.dim()))))

            return hook

        handles = [
            module.register_forward_hook(record(index))
            for index, module in enumerate(vae.decoder)
            if isinstance(module, nn.LayerNorm)
        ]
        try:
            with torch.no_grad():
                vae(latents)
        finally:
            for handle in handles:
                handle.remove()

        spreads = []
        for index in sorted(seen):
            sigma = torch.cat(seen[index])
            spreads.append((sigma.max() / sigma.min()).item())

        assert len(spreads) == 5, spreads
        assert spreads[0] > 1.5, f"the first gain stopped being data-dependent: {spreads[0]:.2f}x"
        assert spreads[-1] < 1.1, f"the last gain stopped being near-constant: {spreads[-1]:.2f}x"
        assert spreads[0] > spreads[-1], spreads


def test_there_is_one_sine_and_sin_still_means_sin_30x():
    """
    ``deep_sdf`` and ``modulated_periodic_activations`` each defined a ``Sine``, and
    ``NSM.models.Sine`` silently meant the hard-coded one (ARCHITECTURE §6). Merging them
    must not move a run: ``get_activation("sin")`` is still ``sin(30 x)``.
    """
    import NSM.models as models
    from NSM.models.deep_sdf import Decoder
    from NSM.models.deep_sdf import Sine as ReExported
    from NSM.models.deep_sdf import get_activation
    from NSM.models.modulated_periodic_activations import Sine

    assert models.Sine is Sine is ReExported
    torch.manual_seed(3)
    x = torch.randn(32)
    assert torch.equal(get_activation("sin")(x), torch.sin(30 * x))
    model = Decoder(latent_size=8, dims=[16, 16], activation="sin").eval()
    with torch.no_grad():
        assert model(torch.randn(5, 11)).shape == (5, 1)
