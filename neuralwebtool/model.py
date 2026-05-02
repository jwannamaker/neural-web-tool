#!/usr/bin/env python
import torch.nn as nn
from torch import Tensor


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
        self.layers = nn.ModuleDict()      # Alternatively use nn.ModuleList()
        self.activations = nn.ModuleDict()

        if len(config["activations"]) != len(layer_sizes) - 1:
            raise ValueError(
                "The number of activation functions must be one less than the number of layers."
            )

        for i in range(len(layer_sizes) - 1):
            layer_name = f"hidden_{i}" if i < len(layer_sizes) - 2 else "output"
            layer_name = "input" if i == 0 else layer_name

            self.layers[layer_name] = nn.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                bias=True,
            )

            self.activations[f"{layer_name}_activate_func"] = self.ACTIVATION_OPTIONS[
                config["activations"][i]
            ]()

    def forward(self, x: Tensor) -> Tensor:
        for layer_name in self.layers.keys():
            x = self.layers[layer_name](x) 
            x = self.activations[f"{layer_name}_activate_func"](x)
        return x
