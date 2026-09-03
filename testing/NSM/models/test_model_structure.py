"""
Structural facts about ``models/`` that no fix in this package may silently change.

Two of them are defects nobody can repair without breaking shipped checkpoints, so the
only thing standing between them and an accidental "cleanup" is an assertion that says
what is true today and why it has to stay that way.
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
    How far the decoder is from affine, and the value scale to read it against.

    An affine map commutes with an affine combination of its inputs. The final ``Tanh``
    does not and is not the question, so it is swapped for ``Identity`` for the duration:
    what is being measured is whether the *stack* supplies any nonlinearity of its own.
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
    ``VAEDecoder.__init__`` used to build ``activation = activation_fn()`` and never append
    it, while the two lines above it appended -- a leaked loop variable, present from the
    first triplanar commit onwards. The dead argument is deleted (#20) and a working
    ``conv_activation`` replaces it (see :class:`TestTheOptInConvActivation`), but **the
    default is still no activation**, because that is the architecture every existing
    checkpoint was fitted as.

    So the stack these assertions describe is what you get by default and what every
    shipped model is: ``ConvTranspose2d -> norm`` x N then ``Conv2d -> Tanh``.
    ``docs/ARCHITECTURE.md`` section 7.1 holds the full account.
    """

    def test_no_pointwise_activation_is_registered_by_default(self):
        assert (
            conv_stack_activations(build_vae()) == []
        ), "the DEFAULT gained an activation; every existing checkpoint stops loading"

    def test_the_only_nonlinearity_is_the_final_tanh(self):
        final = build_vae().decoder[-1]
        assert isinstance(final[0], nn.Conv2d) and isinstance(final[1], nn.Tanh)

    @pytest.mark.parametrize(
        "overrides, affine",
        [
            ({"norm": True, "norm_type": "batch"}, True),
            ({"norm": False}, True),
            ({"norm": True, "norm_type": "layer"}, False),
        ],
        ids=["batch", "no-norm", "layer"],
    )
    def test_the_stack_is_affine_except_where_layernorm_saves_it(self, overrides, affine):
        """
        ARCHITECTURE section 7.1's table, recomputed rather than transcribed.

        ``"batch"`` is the constructor default and evaluates affine, so a five-layer conv
        stack collapses to one. ``"layer"`` -- what both shipped models use -- is nonlinear
        only because LayerNorm divides by a standard deviation computed from its own input.
        The production models work by accident, and the accident is what this pins.
        """
        error, scale = additivity_error(build_vae(**overrides))
        relative = error / scale
        if affine:
            assert relative < 1e-6, f"stopped being affine: {relative:.2e} of value scale"
        else:
            assert relative > 1e-2, f"LayerNorm stopped supplying the nonlinearity: {relative:.2e}"

    def test_the_activation_argument_is_gone(self):
        """
        It was accepted and never read: ``relu`` and ``leakyrelu`` built the same module
        and computed the same numbers, so deleting it changed nothing (#20's rule -- the
        fix for an ignored argument is deletion, never making it authoritative).

        ``VAEDecoder`` takes no ``**kwargs``, so passing it now raises on its own.
        """
        with pytest.raises(TypeError, match="activation"):
            build_vae(activation="relu")


class TestTheOptInConvActivation:
    """
    ``conv_activation`` is the repair for the leaked loop variable, and it is opt-in for a
    structural reason rather than a cautious one: ``nn.Sequential`` names its children by
    position, so inserting a parameterless activation renumbers every later key. The
    default therefore has to remain "no activation" for as long as any pre-Aug-2026
    checkpoint exists -- which is forever.

    ``loader`` requires the config to state which architecture it means; these assert that
    both are real and that the boundary between them is exactly where it should be.
    """

    def test_the_default_is_byte_for_byte_the_historical_architecture(self):
        torch.manual_seed(6)
        historical = build_vae()
        torch.manual_seed(6)
        explicit = build_vae(conv_activation=None)

        assert list(historical.state_dict()) == list(explicit.state_dict())
        x = torch.randn(2, LATENT)
        with torch.no_grad():
            assert torch.equal(historical(x), explicit(x))

    def test_a_pre_existing_checkpoint_loads_at_the_default_and_not_otherwise(self):
        """
        The whole reason this is a flag and not a fix. Same weights, same config, one key
        different -- and the second model cannot be given them.
        """
        torch.manual_seed(7)
        checkpoint = build_vae().state_dict()

        torch.manual_seed(7)
        build_vae(conv_activation=None).load_state_dict(checkpoint, strict=True)

        with pytest.raises(RuntimeError, match="Missing key"):
            build_vae(conv_activation="leaky_relu").load_state_dict(checkpoint, strict=True)

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "swish", "elu"])
    def test_an_activation_is_appended_once_per_block_and_forwards(self, activation):
        vae = build_vae(hidden_dims=[8, 8, 8], conv_activation=activation)
        assert len(conv_stack_activations(vae)) == 3, [type(m).__name__ for m in vae.decoder]
        with torch.no_grad():
            assert vae(torch.randn(2, LATENT)).shape[1] == vae.out_features

    @pytest.mark.parametrize("norm_type,norm", [("layer", "LayerNorm"), ("batch", "BatchNorm2d")])
    def test_it_goes_after_the_norm(self, norm_type, norm):
        """
        Placement is ``conv -> norm -> activation`` and is provisional -- which of the two
        orderings is right is part of what the retrain settles
        (``NSM_TRAINING_IDEAS.md`` Idea 13). Pinned so that changing it is a decision
        someone makes, not a diff someone lands.

        Both norm types, stated rather than defaulted: this asserted the placement through
        whatever the signature default happened to be, so moving that default from
        ``"batch"`` to ``"layer"`` in v0.3.0 turned it red for a reason that had nothing to
        do with placement.
        """
        vae = build_vae(conv_activation="leaky_relu", norm_type=norm_type)
        order = [type(m).__name__ for m in vae.decoder[:3]]
        assert order == ["ConvTranspose2d", norm, "LeakyReLU"], order

    def test_an_unknown_activation_is_refused_by_name(self):
        """
        The deleted argument raised ``UnboundLocalError`` from a half-assigned local for
        anything outside its two-value vocabulary. This one goes through ``get_activation``,
        so ``models/`` has one activation vocabulary rather than two.
        """
        with pytest.raises(ValueError, match="Unknown activation"):
            build_vae(conv_activation="not_an_activation")

    def test_linear_is_refused_and_points_at_none(self):
        """``get_activation('linear')`` returns ``None``, which would silently mean the
        historical stack under a name that reads like a choice."""
        with pytest.raises(ValueError, match="None"):
            build_vae(conv_activation="linear")


class TestWhatLayerNormActuallySupplies:
    """
    The shipped models are nonlinear only because of LayerNorm (see above), so *what kind*
    of nonlinearity that is decides how much the missing activation costs. Three properties,
    none of them re-derivable by reading, all of them constraining any future fix.

    LayerNorm subtracts a mean and divides by a standard deviation. Only the division is
    nonlinear, and it is a radial projection: it preserves direction and rescales magnitude.
    It cannot zero a feature out, cannot form a decision boundary, cannot make the function
    piecewise. Whatever an activation would add is *selectivity*, and none of it is here.
    """

    def test_normalization_is_over_the_whole_feature_map_not_per_position(self):
        """
        ``normalized_shape`` is the full ``(C, H, W)``, so each sample gets **one** scale
        for its entire feature map. The ConvNeXt convention -- normalizing over channels at
        each spatial position -- would give a per-location gain that the next conv could mix
        into genuine multiplicative interactions across space. This is the weaker of the two
        and is what every shipped model runs.
        """
        norms = [m for m in build_vae(norm_type="layer").decoder if isinstance(m, nn.LayerNorm)]
        assert norms, "the layer variant stopped building LayerNorms"
        assert all(len(m.normalized_shape) == 3 for m in norms), [m.normalized_shape for m in norms]

    def test_the_latent_magnitude_is_not_discarded(self):
        """
        LayerNorm is degree-0 homogeneous -- ``LN(cx) == LN(x)`` -- so a stack whose first
        LayerNorm saw only linear maps would be blind to ``||z||``, and the L2 latent prior
        could shrink latents at no reconstruction cost.

        That is not this stack: ``fc`` and the first ``ConvTranspose2d`` both carry biases,
        which break the homogeneity before the first LayerNorm sees anything. Asserted
        rather than assumed, because the conclusions that follow from the homogeneous case
        (an inert latent-norm penalty; interpolating on the sphere rather than the line) are
        wrong here, and are the kind of thing a reader will otherwise derive from theory.
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
        How much nonlinearity LayerNorm actually contributes is how much its per-sample
        sigma *moves* across inputs -- a sigma that never changes is a fixed affine map
        wearing a normalization layer's name.

        Measured here across latents spanning a 2.5x range of norms, matching the fitted
        production range (median ~7.3, bound 10; ``NSM_TRAINING_IDEAS.md`` Idea 4).
        The spread is real at the first LayerNorm and decays towards 1.0 by the last, so
        the deeper layers are close to fixed affine maps. On the shipped 647 model the same
        sweep gives 1.71x, 1.30x, 1.15x, 1.02x, 1.00x.

        Asserted as *first > last* and not as values: the magnitudes depend on width and
        depth, the ordering is the property.
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


class TestOneSine:
    """
    ``deep_sdf`` and ``modulated_periodic_activations`` each defined a ``Sine`` with
    incompatible defaults -- ``w0`` hardcoded to 30 in one, an argument defaulting to 1.0
    in the other -- and ``NSM.models.__init__``'s ``from .deep_sdf import *`` runs before
    the explicit imports, so ``NSM.models.Sine`` silently meant the hardcoded one
    (ARCHITECTURE section 6). The parameterized one won; it moved into ``deep_sdf`` when
    ``modulated_periodic_activations`` was retired with the ``implicit`` model type.
    """

    def test_there_is_only_one_sine(self):
        import NSM.models as models
        from NSM.models.deep_sdf import Sine

        assert models.Sine is Sine

    def test_the_sin_activation_still_computes_sin_30x(self):
        """
        What makes merging the two safe: no run's arithmetic changes. The deleted class
        computed ``torch.sin(30 * input)`` with 30 inlined; ``get_activation`` now returns
        ``Sine(w0=30)``, and the parameterized default of 1.0 must not leak in here.
        """
        from NSM.models.deep_sdf import get_activation

        torch.manual_seed(3)
        x = torch.randn(32)
        assert torch.equal(get_activation("sin")(x), torch.sin(30 * x))

    def test_a_sine_decoder_forwards(self):
        """The activation reached through a real ``Decoder``, not just constructed."""
        from NSM.models.deep_sdf import Decoder

        torch.manual_seed(4)
        model = Decoder(latent_size=8, dims=[16, 16], activation="sin").eval()
        with torch.no_grad():
            assert model(torch.randn(5, 11)).shape == (5, 1)
