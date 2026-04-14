#!/usr/bin/env python
import torch
import torch.nn as nn
from torch import optim

LOSS_OPTIONS = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "nll_loss": nn.NLLLoss(),
}

OPTIMIZER_OPTIONS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.criterion = LOSS_OPTIONS[config["loss"]]
        self.optimizer = OPTIMIZER_OPTIONS[config["optimizer"]](
            model.parameters(), lr=config["lr"]
        )
