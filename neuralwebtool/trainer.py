#!/usr/bin/env python
import torch.nn as nn
from torch import optim


class Trainer:
    LOSS_OPTIONS = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse_loss": nn.MSELoss(),
        "nll_loss": nn.NLLLoss(),
    }

    OPTIMIZER_OPTIONS = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }

    def __init__(self, model: nn.Module, config: dict):
        """
        Args:
            model (nn.Module): The model to be trained
            config (dict): A dictionary containing the training configuration, including:
                - "loss": The loss function to use (e.g., "cross_entropy", "mse_loss", "nll_loss")
                - "optimizer": The optimizer to use (e.g., "adam", "sgd")
                - "lr": The learning rate for the optimizer
        """
        self.model = model
        self.criterion = Trainer.LOSS_OPTIONS[config["loss"]]
        self.optimizer = Trainer.OPTIMIZER_OPTIONS[config["optimizer"]](model.parameters(), lr=config["lr"])

    def train_step(self, images, labels):
        self.optimizer.zero_grad()              # Reset the gradients, by default they accumulate
        output = self.model(images)             # Forward Pass
        loss = self.criterion(output, labels)   # Calculate the accuracy of the model by comparing truth and predictions
        loss.backward()                         # Backprop, compute gradients
        self.optimizer.step()                   # Gradient descent, here we update the weights of model
        return loss.item()
