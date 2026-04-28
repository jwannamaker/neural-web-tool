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
        self.optimizer = Trainer.OPTIMIZER_OPTIONS[config["optimizer"]](
            model.parameters(), lr=config["lr"])

    def train_step(self, images, labels):
        # Reset the gradients, by default they accumulate
        self.optimizer.zero_grad()
        output = self.model(images)             # Forward Pass
        
        # Calculate the loss depending on the criterion
        import torch.nn.functional as F
        if isinstance(self.criterion, nn.MSELoss):
            labels_one_hot = F.one_hot(labels, num_classes=output.shape[1]).float()
            loss = self.criterion(output, labels_one_hot)
        elif isinstance(self.criterion, nn.NLLLoss):
            log_probs = F.log_softmax(output, dim=1)
            loss = self.criterion(log_probs, labels)
        else:
            loss = self.criterion(output, labels)
            
        loss.backward()                         # Backprop, compute gradients
        # Gradient descent, here we update the weights of model
        self.optimizer.step()
