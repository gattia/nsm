"""DeepSDF: an MLP that maps ``[latent, xyz]`` to one signed distance per object.

The original architecture NSM was built on, and still the ``model_type: deepsdf``
path and the second half of ``two_stage``. :class:`Decoder` carries the whole of it;
``loader.load_model`` translates config keys into its parameters (the two vocabularies
differ -- ``layer_latent_in`` is ``latent_in``, ``layer_dimensions`` is ``dims``).

Two things here are load-bearing and not obvious. ``SIREN_W0`` is the frequency scale
``activation='sin'`` initialises against, and it must match the ``Sine`` that
:func:`get_activation` returns -- there were two ``Sine`` classes computing the same
thing until Aug 2026 (``docs/ARCHITECTURE.md`` section 6). ``PROGRESSIVE_PARAMS``
schedules Curriculum-DeepSDF's depth phase-in by *epoch*, so a forward pass under
``progressive_add_depth`` is not a pure function of its input.
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulated_periodic_activations import Sine

logger = logging.getLogger(__name__)

#: SIREN's frequency scaling. See the paper's sec. 3.2, final paragraph, and supplement
#: sec. 1.5, for the discussion of the factor 30.
SIREN_W0 = 30

PROGRESSIVE_PARAMS = {
    "n_layers": 3,
    "layers": {
        5: {  # Or -3, -2, -1....
            "start_epoch": 200,
            "warmup_epochs": 200,
        },
        6: {  # Or -3, -2, -1....
            "start_epoch": 600,
            "warmup_epochs": 200,
        },
        7: {  # Or -3, -2, -1....
            "start_epoch": 1010,
            "warmup_epochs": 200,
        },
    },
}


class Decoder(nn.Module):
    """MLP decoder: ``[latent, xyz] -> sdf`` per object, optionally with skips.

    The layer stack is ``dims`` widened by ``latent_size + 3`` at the input. Three
    knobs change its *shape* rather than its size, and each is a different mechanism:

    * ``latent_in`` repeats the input vector into the named layers. Under
      ``concat_latent_input=False`` the layer is *narrowed* so the concatenation
      restores its declared width; under ``True`` it is widened instead. Same key,
      opposite arithmetic -- see :meth:`get_layer_dims`.
    * ``layer_split`` gives each object its own tail from that layer on, so
      ``n_objects > 1`` is either one shared trunk with an ``n_objects``-wide output
      or ``n_objects`` separate tails. ``False`` is coerced to ``None`` on purpose:
      ``False == 0`` made "off" indistinguishable from a split at layer 0 (#46).
    * ``progressive_add_depth`` phases later blocks in over training, which makes
      ``forward`` depend on ``epoch``; it raises if the epoch is not supplied.

    ``**kwargs`` exists only to refuse or warn on parameters that were accepted and
    never read (``xyz_in_all``, ``latent_noise_sigma``, ``norm_layers``,
    ``latent_dropout``); it is not an extension point.
    """

    def __init__(
        self,
        latent_size,
        dims,
        n_objects=1,
        dropout=None,
        dropout_prob=0.2,
        latent_in=(),
        weight_norm=True,
        activation="relu",  # "relu" or "sin"
        final_activation="tanh",  # "sin", "linear"
        concat_latent_input=False,
        progressive_add_depth=False,
        progressive_depth_params=PROGRESSIVE_PARAMS,
        layer_split=None,
        **kwargs,
    ):
        """
        latent_size (int): size of the latent input vector to the decoder network
        dims (list of ints): list containing the size of each layer in MLP.
        n_objects (int): number of objects to predict
        dropout (list of ints): where to apply dropout to the encoder
        dropout_prob (float) : probability with which dropout is applied
        latent_in (list of ints): where to repeat the latent vector in the decoder
        weight_norm (bool): whether to apply weight normalization
        """
        super(Decoder, self).__init__()

        if "latent_dropout" in kwargs:
            warnings.warn(
                "latent_dropout is deprecated and ignored: the latent-input dropout it once "
                "enabled no longer exists, and per-layer dropout (dropout + dropout_prob) is "
                "a different mechanism, not a replacement. Delete the argument.",
                DeprecationWarning,
            )

        # Deleted arguments a config on disk can still carry. Refused when truthy, ignored
        # when falsy -- falsy is what every NSM-owned config ships, and it asked for nothing.
        if kwargs.get("xyz_in_all"):
            raise TypeError(
                "xyz_in_all was never implemented: no NSM decoder injects xyz at each "
                "layer. Decoder accepted the argument, documented it, and never read it. "
                "Delete the argument (or the config key); there is no replacement."
            )

        if kwargs.get("latent_noise_sigma"):
            raise TypeError(
                "latent_noise_sigma was never implemented: Decoder stored it and forward "
                "never read it, so no run has ever had latent noise added. Delete the "
                "argument; there is no replacement."
            )

        # norm_layers is not simply inert, so the two cases are answered differently. The
        # branch that built the LayerNorms is an `elif` under weight_norm (01d774a,
        # Jun 2023), so:
        #   weight_norm on  -> nothing was ever built; the key is provably a no-op and
        #                      refusing it would break configs the defect never touched.
        #   weight_norm off -> LayerNorms were built and used, and the checkpoint carries
        #                      bn.* keys. That architecture is gone, so say so.
        if kwargs.get("norm_layers"):
            if weight_norm:
                logger.warning(
                    "norm_layers (config key: layers_with_norm) is set but has no effect "
                    "and never had one under weight_norm=True: the branch that built the "
                    "norm layers was unreachable. The model is unchanged; delete the key."
                )
            else:
                raise TypeError(
                    "norm_layers (config key: layers_with_norm) is gone, and with "
                    "weight_norm=False it did something: LayerNorm was applied at those "
                    "layers and the checkpoint carries bn.* keys. That architecture can no "
                    "longer be built here -- pin NSM < 0.3.0 to load such a checkpoint. "
                    "Under weight_norm=True the key was always a no-op."
                )

        self._activation_ = activation
        self._final_activation_ = final_activation
        self.concat_latent_input = concat_latent_input
        # list(), not `+ dims`: TwoStageDecoder's default mlp_params carries a tuple, and
        # `[x] + (...)` is a TypeError -- so the type was not constructible at all (#46).
        self.dims = [latent_size + 3] + list(dims)
        self.latent_in = latent_in
        self.progressive_add_depth = progressive_add_depth
        self.progressive_depth_params = progressive_depth_params
        # `is`, not `==`: `False == 0` makes default_config.json's "off" indistinguishable
        # by value from a deliberate split at layer 0 (#46; KNOWN_ISSUES History 14).
        self.layer_split = None if layer_split is False else layer_split
        self.n_objects = n_objects

        # layers:
        # 0: input
        # 1 to N-hidden: NN
        # -1: output
        if (n_objects == 1) or ((n_objects > 1) and (self.layer_split is not None)):
            self.dims = self.dims + [1]
        else:
            self.dims = self.dims + [n_objects]

        self.layers = nn.ModuleList()

        # Add the rest of the layers
        for layer in range(len(self.dims) - 1):
            # get layer input and output dimensions
            in_dim, out_dim = self.get_layer_dims(layer)
            if self.layer_split is not None and layer >= self.layer_split:
                lin_layer = nn.ModuleList()
                for _ in range(self.n_objects):
                    lin_layer.append(
                        self.lin_layer_(
                            in_dim=in_dim, out_dim=out_dim, layer=layer, weight_norm=weight_norm
                        )
                    )
            else:
                lin_layer = self.lin_layer_(
                    in_dim=in_dim, out_dim=out_dim, layer=layer, weight_norm=weight_norm
                )
            self.layers.append(lin_layer)

        self.activation = get_activation(self._activation_)
        self.final_activation = get_activation(self._final_activation_)

        if self.activation is None:
            raise ValueError(
                f"activation={self._activation_!r} is not a hidden-layer activation: "
                f"get_activation returns None for it, and forward would call None. "
                f"'linear' is supported in the final position only "
                f"(final_activation='linear'), where forward guards for it."
            )

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.epoch = None

    def lin_layer_(self, in_dim, out_dim, layer, weight_norm):
        """One ``nn.Linear``, initialised for ``self._activation_`` and optionally
        weight-normed. ``layer == 0`` selects SIREN's first-layer bound, which differs
        from the rest by a factor of ``sqrt(6 / n) / SIREN_W0``."""
        lin_layer = nn.Linear(in_dim, out_dim)
        # initialize the weights - particularly for the sine activation
        init_weights(module=lin_layer, activation=self._activation_, first_layer=layer == 0)
        # add weight norm if specified
        if weight_norm is True:
            lin_layer = nn.utils.weight_norm(lin_layer)
        return lin_layer

    def get_layer_dims(self, layer):
        """Input and output width of ``layer``, given how the latent is re-injected.

        ``concat_latent_input`` decides which side of the layer ``latent_in`` pays for.
        ``False``: the *output* of the preceding layer is narrowed by ``dims[0]`` so the
        concatenation brings it back to the declared width. ``True``: the *input* is
        widened by ``dims[0]`` and the declared widths stand. Anything else (the
        ``None`` a config can carry) takes neither branch and returns the plain dims.
        """
        if self.concat_latent_input is False:
            in_dim = self.dims[layer]
            if layer + 1 in self.latent_in:
                out_dim = self.dims[layer + 1] - self.dims[0]
            else:
                out_dim = self.dims[layer + 1]
        elif self.concat_latent_input is True:
            out_dim = self.dims[layer + 1]
            if layer in self.latent_in:
                in_dim = self.dims[layer] + self.dims[0]
            else:
                in_dim = self.dims[layer]
        else:
            in_dim = self.dims[layer]
            out_dim = self.dims[layer + 1]

        return in_dim, out_dim

    def forward_branch_(self, x, input_, layer, layer_idx):
        """One block: optional latent concat, the layer, then activation and dropout.

        Activation and dropout are applied to hidden layers only -- the final layer's
        nonlinearity is ``final_activation``, applied once in :meth:`forward`. Under
        ``progressive_add_depth`` a block that has not started yet returns ``x``
        unchanged, so a phased-in block must be hidden-to-hidden: the skip is an
        identity and cannot change width (#46)."""
        if layer_idx in self.latent_in:
            xi = torch.cat([x, input_], 1)
        else:
            xi = x

        if (self.progressive_add_depth is True) and (
            layer_idx in self.progressive_depth_params["layers"]
        ):
            if self.epoch >= self.progressive_depth_params["layers"][layer_idx]["start_epoch"]:
                x = self.progressive_layer(xi, layer, layer_idx)
            else:
                # Not started yet: skip the block, which is what `progressive_layer`
                # computes at zero weight one epoch later. The skip is an identity, so a
                # phased-in block has to be hidden-to-hidden (#46).
                return x
        else:
            x = layer(xi)

        # only apply normalization/ regular activation to
        # hidden layers (not output)
        if layer_idx < len(self.layers) - 1:
            x = self.activation(x)

            if (
                self.dropout is not None and layer_idx in self.dropout
            ):  # and (self._activation_ != "sin")
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return x

    # input: N x (L+3)
    def forward(self, input_, epoch=None):  # noqa: D102 - see the class docstring
        # Assign the epoch in case needed (for progressive depth)
        if epoch is not None:
            self.epoch = epoch

        if self.progressive_add_depth is True and self.epoch is None:
            raise ValueError(
                "progressive_add_depth needs the epoch to know which blocks have started: "
                "call forward(input_, epoch=<int>). Without it the comparison against "
                "start_epoch is None >= int."
            )

        x = input_

        for layer_idx, layer in enumerate(self.layers):  # range(0, self.num_layers - 1):
            if self.layer_split is not None and layer_idx >= self.layer_split:

                if layer_idx == self.layer_split:
                    x = [
                        x,
                    ] * self.n_objects

                for i in range(self.n_objects):
                    x_ = self.forward_branch_(
                        x=x[i], input_=input_, layer=layer[i], layer_idx=layer_idx
                    )
                    if x_ is None:
                        continue
                    else:
                        x[i] = x_
            else:
                x = self.forward_branch_(x=x, input_=input_, layer=layer, layer_idx=layer_idx)

            if x is None:
                continue

        if self.layer_split is not None:
            x = torch.cat(x, dim=1)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def progressive_layer(self, xi, layer, layer_idx):
        """Blend ``xi`` with ``layer(xi)`` over the block's warmup (Curriculum-DeepSDF).

        Weight is ``((epoch - start) / warmup) ** 2`` on the new block and ``1 - that``
        on the identity, so the block arrives at zero influence and reaches full weight
        at ``start + warmup``. One start condition, not three: ``epoch == start`` used
        to fall through to full weight (``KNOWN_ISSUES`` § History 14).
        """
        # use this as a way to store the progress of the network so far.
        # this way if we try to use the partly tuned model for inference, it will be able to
        # use the weights that have been tuned so far.

        # progresive tuning of latter layers is from Curriculum DeepSDF
        # code was adapted from:
        # https://github.com/haidongz-usc/Curriculum-DeepSDF
        start = self.progressive_depth_params["layers"][layer_idx]["start_epoch"]
        warmup = self.progressive_depth_params["layers"][layer_idx]["warmup_epochs"]
        end = start + warmup
        # One start condition, not three. `start < self.epoch < end` excluded epoch ==
        # start, which then fell through to full weight (KNOWN_ISSUES History 14), and the
        # `epoch < start` RuntimeError it replaced was unreachable from the one caller.
        if self.epoch < end:
            # during warmup... linearly phase this block in
            # https://github.com/haidongz-usc/Curriculum-DeepSDF/blob/ca216dda8edc6435139a6f657c45800791be94a7/networks/deep_sdf_decoder_train.py#L113
            new_weight = (self.epoch - start) / warmup
            new_weight = new_weight**2
            base_weight = 1 - new_weight

            x_base = xi * base_weight
            x_new = layer(xi) * new_weight
            x = x_base + x_new
        else:
            # after start + warmup epochs just apply this block as a normal layer
            x = layer(xi)

        return x


def init_weights(module, activation, first_layer=False):
    """
    Initializes the weights of a linear layer based on the activation function.
    """
    if isinstance(module, nn.Linear):
        num_input = module.weight.size(-1)
        if activation == "sin":
            with torch.no_grad():
                if first_layer is True:
                    b = 1 / num_input
                elif first_layer is False:
                    b = np.sqrt(6 / num_input) / SIREN_W0

                torch.nn.init.uniform_(module.weight, -b, b)


def get_activation(activation):
    """The ``nn.Module`` for an activation name, or ``None`` for ``"linear"``.

    ``None`` is a real answer, not a failure: ``linear`` is supported in the *final*
    position, where ``forward`` guards for it. Asking for it as a hidden-layer
    activation is refused in ``Decoder.__init__`` rather than here, because here it
    is indistinguishable from the legitimate case.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "swish":
        return nn.SiLU()
    elif activation == "sin":
        # One Sine, imported rather than redefined here -- ARCHITECTURE.md section 6 for
        # why the duplicate was invisible. Both computed sin(30 * x).
        return Sine(w0=SIREN_W0)
    elif activation == "linear":
        return None
    else:
        raise ValueError(f"Unknown activation function: {activation}")
