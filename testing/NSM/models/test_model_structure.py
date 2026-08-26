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


class TestTheVAEHasNoActivation:
    """
    ``VAEDecoder.__init__`` used to build ``activation = activation_fn()`` and never append
    it, while the two lines above it appended -- a leaked loop variable. The argument is
    deleted (#20); the shape it left behind is not. The stack is ``ConvTranspose2d -> norm``
    x N then ``Conv2d -> Tanh``; ``LeakyReLU`` appears nowhere.

    **It cannot be added unconditionally, only opt-in.** Inserting the activations shifts
    every later module's index inside ``nn.Sequential``, so all three shipped checkpoints
    stop loading -- and the weights were fitted without them regardless, so remapping the
    keys would load a model that computes something else. A ``conv_activation`` flag
    defaulting to off builds the identical module list and changes nothing, which is what
    makes the fix available at all; what it needs beyond that is a retrain to show it is
    worth having. ``docs/ARCHITECTURE.md`` section 7.1 holds the full account; these
    assertions are what stops someone closing the gap by reflex.
    """

    def test_no_pointwise_activation_is_registered_in_the_conv_stack(self):
        activations = [
            module
            for module in build_vae().decoder[:-1].modules()
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU))
        ]
        assert activations == [], "an activation was added; every shipped checkpoint moves"

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


class TestOneSine:
    """
    ``deep_sdf`` and ``modulated_periodic_activations`` each define a ``Sine``, with
    incompatible defaults -- ``w0`` hardcoded to 30 in one, an argument defaulting to 1.0
    in the other -- and ``NSM.models.__init__``'s ``from .deep_sdf import *`` runs first,
    so ``NSM.models.Sine`` silently means the hardcoded one (ARCHITECTURE section 6).
    """

    def test_the_two_compute_the_same_function_at_w0_30(self):
        """What makes merging them safe: no run's arithmetic changes."""
        from NSM.models.deep_sdf import Sine as HardcodedSine
        from NSM.models.modulated_periodic_activations import Sine as ParameterizedSine

        torch.manual_seed(3)
        x = torch.randn(32)
        assert torch.equal(HardcodedSine()(x), ParameterizedSine(30)(x))

    def test_the_hardcoded_sines_initializer_never_runs(self):
        """``def __init(self)`` -- one underscore short, and name-mangled to ``_Sine__init``."""
        from NSM.models.deep_sdf import Sine as HardcodedSine

        assert "_Sine__init" in vars(HardcodedSine)
        assert "__init__" not in vars(HardcodedSine)

    @pytest.mark.xfail(
        strict=True,
        reason="#46/ARCHITECTURE section 6: two Sine classes, and the star-import picks one",
    )
    def test_there_is_only_one_sine(self):
        import NSM.models as models
        from NSM.models.modulated_periodic_activations import Sine

        assert models.Sine is Sine
