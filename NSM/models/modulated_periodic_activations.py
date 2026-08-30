"""SIREN-style blocks and the modulated implicit decoder built from them.

``model_type: implicit`` in ``loader``. Two families of block share one interface --
:class:`LinearBlock` (ReLU by default) and :class:`SirenLinear` (sine, with the
initialisation the activation requires) -- and a *factory* per family lets
:class:`MLP` build a stack without knowing which. :class:`ImplicitDecoder` then wires
the latent in one of two ways, and they are architecturally different rather than
two settings of one thing:

* ``modulation=False`` concatenates the latent onto ``xyz`` at the input, so the MLP
  input width is ``3 + latent_dim``.
* ``modulation=True`` feeds the latent through a :class:`ModulationNetwork` and
  multiplies its per-layer outputs into the MLP's hidden activations, so the MLP
  input width is 3 and the latent never enters as a coordinate.

Adapted from https://github.com/kklemon/gon-pytorch.
"""

from functools import partial
from math import pi, sqrt
from typing import Callable, List, Optional

import torch
from torch import nn


# https://github.com/kklemon/gon-pytorch/blob/master/gon_pytorch/modules.py
class LinearBlock(nn.Module):
    """Weight-normed linear layer plus an activation, unless it is the last block.

    ``forward`` is ``activation(linear(x))``, or ``linear(x)`` when ``is_last``.
    ``is_first`` carries no behaviour here -- it is recorded for subclasses whose
    initialisation depends on it (see :class:`SirenLinear`).
    """

    def __init__(
        self,
        in_features,
        out_features,
        linear_cls,
        activation=nn.ReLU,
        bias=True,
        is_first=False,
        is_last=False,
    ):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.linear = nn.utils.weight_norm(linear_cls(in_features, out_features, bias=bias))
        self.bias = bias
        self.is_first = is_first
        self.is_last = is_last
        self.activation = None if is_last else activation()

    def forward(self, x):  # noqa: D102 - see the class docstring
        x = self.linear(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

    def __repr__(self):
        return (
            f"LinearBlock(in_features={self.in_f}, out_features={self.out_f}, linear_cls={self.linear}, "
            f"activation={self.activation}, bias={self.bias}, is_first={self.is_first}, is_last={self.is_last})"
        )


class Sine(nn.Module):
    """``sin(w0 * x)``. The one ``Sine`` in NSM.

    ``deep_sdf`` imports this rather than defining its own; there were two identical
    implementations until Aug 2026 and neither referenced the other, so a change to
    either would have applied to half the models (``docs/ARCHITECTURE.md`` section 6).
    ``w0`` scales the input frequency; SIREN's 30 is ``deep_sdf.SIREN_W0``, and the
    weight initialisation in :meth:`SirenLinear.init_weights` is derived from it, so
    the two have to move together.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):  # noqa: D102 - see the class docstring
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"Sine(w0={self.w0})"


class SirenLinear(LinearBlock):
    """A :class:`LinearBlock` whose activation is :class:`Sine` and whose weights are
    initialised for it.

    The initialisation is not optional -- SIREN diverges without it -- so it runs in
    ``__init__`` rather than being left to the caller. See :meth:`init_weights` for
    the bounds.
    """

    def __init__(
        self,
        in_features,
        out_features,
        linear_cls=nn.Linear,
        w0=30,
        bias=True,
        is_first=False,
        is_last=False,
    ):
        super().__init__(
            in_features, out_features, linear_cls, partial(Sine, w0), bias, is_first, is_last
        )
        self.w0 = w0
        self.init_weights()

    def init_weights(self):
        """Uniform weights and biases in SIREN's bounds: ``1 / in_f`` for the first
        layer, ``sqrt(6 / in_f) / w0`` after it (Sitzmann et al., sec. 3.2). The first
        layer differs because its input is coordinates rather than a previous layer's
        already-scaled activations."""
        if self.is_first:
            b = 1 / self.in_f
        else:
            b = sqrt(6 / self.in_f) / self.w0

        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)


class BaseBlockFactory:
    """Interface for the callables :class:`MLP` builds its stack from.

    A factory rather than a class reference because a block needs constructor
    arguments (``w0``, ``linear_cls``, ``bias``) that :class:`MLP` does not know and
    should not have to forward.
    """

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        """Build one block of the given widths. ``is_first``/``is_last`` select the
        initialisation and whether an activation is appended."""
        raise NotImplementedError


class LinearBlockFactory(BaseBlockFactory):
    """Builds :class:`LinearBlock`\\ s with a fixed linear class and activation."""

    def __init__(self, linear_cls=nn.Linear, activation_cls=nn.ReLU, bias=True):
        self.linear_cls = linear_cls
        self.activation_cls = activation_cls
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        """One :class:`LinearBlock`; the activation is dropped when ``is_last``."""
        return LinearBlock(
            in_f, out_f, self.linear_cls, self.activation_cls, self.bias, is_first, is_last
        )


class SirenBlockFactory(BaseBlockFactory):
    """Builds :class:`SirenLinear`\\ s at a fixed ``w0``."""

    def __init__(self, linear_cls=nn.Linear, w0=30, bias=True):
        self.linear_cls = linear_cls
        self.w0 = w0
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        """One :class:`SirenLinear`; ``is_first`` selects SIREN's first-layer bound."""
        return SirenLinear(in_f, out_f, self.linear_cls, self.w0, self.bias, is_first, is_last)


class MLP(nn.Module):
    """``num_layers`` blocks from ``block_factory``, input to output, with dropout
    between the hidden ones.

    ``num_layers`` counts input-to-output inclusive, so 1 is a single block straight
    from ``in_dim`` to ``out_dim``; below 1 raises. ``forward`` takes optional
    ``modulations`` -- one tensor per hidden block, multiplied into that block's
    output -- which is how :class:`ImplicitDecoder` injects a latent without
    concatenating it.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        block_factory: BaseBlockFactory,
        dropout: float = 0.0,
        final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.blocks = nn.ModuleList()

        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1 (input to output); got {self.num_layers}")

        for i in range(self.num_layers):
            in_feat = self.in_dim if i == 0 else self.hidden_dim
            out_feat = self.out_dim if i + 1 == self.num_layers else self.hidden_dim

            is_first = i == 0
            is_last = i + 1 == self.num_layers

            curr_block = [block_factory(in_feat, out_feat, is_first=is_first, is_last=is_last)]
            if not is_last and dropout:
                curr_block.append(nn.Dropout(dropout))

            self.blocks.append(nn.Sequential(*curr_block))

        self.final_activation = final_activation
        if final_activation is None:
            self.final_activation = nn.Identity()

    def forward(self, x, modulations=None):  # noqa: D102 - see the class docstring
        for i, block in enumerate(self.blocks):
            x = block(x)
            if modulations is not None and len(self.blocks) > i + 1:
                x *= modulations[i]
        return self.final_activation(x)


class ModulationNetwork(nn.Module):
    """
    https://github.com/kklemon/gon-pytorch/blob/2e374124cdf4ec57f135fe103e5f7923e07c96c8/gon_pytorch/modules.py#LL312C1-L330C20
    """

    def __init__(self, in_dim: int, mod_dims: List[int], activation=nn.ReLU, weight_norm=True):
        """
        in_dim = latent_dim size
        mod_dims = hidden_dim size from MLP

        At each layer, the output of the previous layer (mod_dims[i-1]) is concatenated with the
        latent (in_dim) — in that order: ``cat([out, input])``, previous-layer features first,
        latent second, which is baked into the weight layout of any trained checkpoint — and
        then fed into the next layer to produce an output of dimension mod_dims[i]. The output
        of each layer is stored/returned in a list. This list of outputs is the same size as the
        hidden layers of the original MLP. This list of outputs is then used to modulate the
        weights of the original MLP (alpha_i in eqn 2. of the paper)

        """
        super().__init__()

        self.blocks = nn.ModuleList()
        for i, mod_dim in enumerate(mod_dims):
            # if layer 0 then there is no previous layer to concatenate with so just use the latent_dim (in_dim)
            # else concatenate the latent_dim (in_dim) with the previous layer's output (mod_dims[i-1])
            lin = nn.Linear(in_dim + (mod_dims[i - 1] if i else 0), mod_dim)
            if weight_norm is True:
                lin = nn.utils.weight_norm(lin)
            self.blocks.append(nn.Sequential(lin, activation()))

    def forward(self, input):  # noqa: D102 - see the class docstring
        out = input
        mods = []
        for block in self.blocks:
            out = block(out)
            # store outputs of the modulation network to be passed to the MLP
            mods.append(out)
            # concatenate the previous layer's output (mod_dims[i-1]) with the latent (in_dim),
            # in that order; this is only for the input to the next layer, not passed to MLP.
            out = torch.cat([out, input], dim=-1)
        return mods


class ImplicitDecoder(nn.Module):
    """Latent + ``xyz`` to SDF, by concatenation or by modulation.

    ``forward`` splits its input as ``[latent, xyz]`` with ``xyz`` the **last three**
    columns, matching :class:`~NSM.models.deep_sdf.Decoder`. Which half of the input
    the latent occupies then depends on ``modulation``, and so does the MLP's input
    width: ``3 + latent_dim`` when concatenating, 3 when modulating. That makes
    ``modulation`` an architecture choice a checkpoint is fixed to, not a runtime
    switch.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        block_factory: BaseBlockFactory,
        #  pos_encoder: CoordinateEncoding = None,
        modulation: bool = False,
        dropout: float = 0.0,
        final_activation=torch.sigmoid,
    ):
        super().__init__()

        # self.pos_encoder = pos_encoder
        self.latent_dim = latent_dim

        self.mod_network = None
        if modulation:
            self.mod_network = ModulationNetwork(
                in_dim=latent_dim,
                mod_dims=[hidden_dim for _ in range(num_layers - 1)],
                activation=nn.ReLU,
            )

        self.net = MLP(
            in_dim=3 + latent_dim * (not modulation),
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            block_factory=block_factory,
            dropout=dropout,
            final_activation=final_activation,
        )

    def forward(self, input_, epoch=None):  # noqa: D102 - see the class docstring
        # if self.pos_encoder is not None:
        #     input = self.pos_encoder(input)

        xyz = input_[:, -3:]
        latent = input_[:, :-3]

        if self.mod_network is None:
            b, *spatial_dims, c = xyz.shape
            latent = latent.view(b, *((1,) * len(spatial_dims)), -1).repeat(1, *spatial_dims, 1)
            out = self.net(torch.cat([latent, xyz], dim=-1))
        else:
            mods = self.mod_network(latent)
            out = self.net(xyz, mods)

        return out
