"""Triplanar and MLP decoders summed: ``sdf = triplanar(z1, xyz) + mlp(z2, xyz)``.

``model_type: two_stage`` in ``loader``. The latent is split in half by position --
the first ``model_latent_size`` columns drive the triplanar branch and the next
``model_latent_size`` the MLP -- so the two halves are not interchangeable and the
split point is fixed by the checkpoint.

The intent is coarse-plus-detail: the triplanar branch carries what a feature grid is
good at and the MLP the high-frequency remainder. Nothing enforces that division;
it is what the sum is expected to learn.
"""

import torch
from torch import nn

from .deep_sdf import Decoder
from .triplanar import TriplanarDecoder

default_triplanar_params = {
    "latent_dim": 256,
    "n_objects": 2,
    "conv_hidden_dims": [512, 512, 512, 512, 512],
    "conv_deep_image_size": 2,
    "conv_norm": True,
    "conv_norm_type": "layer",
    "conv_start_with_mlp": True,
    "conv_activation": None,
    "sdf_latent_size": 128,
    "sdf_hidden_dims": [512, 512, 512],
    "sdf_weight_norm": True,
    "sdf_final_activation": "tanh",
    "sdf_activation": "relu",
    # Stated, not left to TriplanarDecoder's signature: `padding` is not a learned
    # parameter, so a wrong value here loads clean and samples at the wrong scale (#26).
    "padding": 0.1,
}

default_mlp_params = {
    "latent_size": 256,
    "dims": (512, 512, 512, 512, 512, 512, 512, 512),
    "n_objects": 2,
    "dropout": None,
    "dropout_prob": 0.0,
    "latent_in": (),
    "weight_norm": True,
    "activation": "relu",  # "relu" or "sin"
    "final_activation": "tanh",  # "sin", "linear"
    "concat_latent_input": True,
}


class TwoStageDecoder(nn.Module):
    """
    Create a two stage model that takes in a latent vector and 3d coordinates and
    outputs the SDF for each point.

    It takes 1/2 of the latent vector and passes it through a triplanar decoder
    It takes the other 1/2 of the latent vector and passes it through an MLP

    These outputs both predict the SDF for each points, the outputs are then summed
    and returned as the final SDF prediction.
    """

    def __init__(
        self,
        latent_size=512,
        n_objects=2,
        triplanar_params: dict = default_triplanar_params,
        mlp_params: dict = default_mlp_params,
    ):
        super(TwoStageDecoder, self).__init__()

        self.latent_size = latent_size
        self.model_latent_size = latent_size // 2
        assert latent_size % 2 == 0, "latent_size must be even"

        self.n_objects = n_objects

        # Copied before they are written to: both defaults are module-level dicts, so one
        # construction used to rewrite what every later default construction meant (#46).
        triplanar_params = dict(triplanar_params)
        mlp_params = dict(mlp_params)

        triplanar_params["latent_dim"] = self.model_latent_size
        triplanar_params["n_objects"] = self.n_objects
        mlp_params["latent_size"] = self.model_latent_size
        mlp_params["n_objects"] = self.n_objects

        self.triplanar_params = triplanar_params
        self.mlp_params = mlp_params

        self.triplanar = TriplanarDecoder(**triplanar_params)
        self.mlp = Decoder(**mlp_params)

    def forward(self, input, epoch=None):  # noqa: D102 - see the class docstring
        # Split the latent vector in half
        latent_triplanar = input[:, : self.model_latent_size]
        latent_mlp = input[:, self.model_latent_size : self.model_latent_size * 2]

        # get the xyz coordinates
        xyz = input[:, -3:]

        # Pass the latent vector  & xyz through the triplanar decoder
        sdf_triplanar = self.triplanar(torch.cat([latent_triplanar, xyz], dim=-1))
        # Pass the other half of the latent vector & xyz through the MLP
        sdf_mlp = self.mlp(torch.cat([latent_mlp, xyz], dim=-1))

        # Sum the outputs
        sdf = sdf_triplanar + sdf_mlp

        return sdf
