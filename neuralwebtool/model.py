#!/usr/bin/env python
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class Network(nn.Module):
    """
    TODO: Use these options for the different activation functions a user can choose from, perhaps in a dropdown menu.

    Reasonable default config:
    config={
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
            "activations": ["relu", "relu", "linear"],
        },
    """

    ACTIVATION_OPTIONS = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "linear": nn.Identity,
    }

    def __init__(
        self,
        layer_sizes=[784, 128, 64, 10],
        config={
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
            "activations": ["relu", "relu", "linear"],
        },
    ):
        """
        Creating the layers that will apply a linear transformation to the incoming data
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): Whether the layer will learn an additive bias or not.

        Shape:
            Input: (*, in_features) where * can be None.
            Output: (*, out_features) where * is the same as input

        Attributes:
            weight: Learnable weights of the module with shape (in_features, out_features) and initialized by uniform(-k,k) where k=1/sqrt(in_features)
            bias: Learnable bias of the module with shape (out_features) and initialized by uniform(-k,k) where k=1/sqrt(in_features)
        """
        super(Network, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    bias=True,
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )

        activations = config["activations"]

        

    def forward(self, input: Tensor):  # -> Tensor:
        """
        TODO: Accept different choices as the activation function.
        TODO: Test that len(activations) == number of layers
        """
        pass
