#!/usr/bin/env python
"""Command-line interface for neural network training and experimentation."""
import click
import torch

from neuralwebtool.model import Network
from neuralwebtool.trainer import Trainer
from neuralwebtool.data import Data


@click.group()
def cli() -> None:
    """Neural Web Tool - Train and test neural networks from the command line."""
    pass


@cli.command()
@click.option('--layers', default='784,128,64,10', help='Comma-separated layer sizes')
@click.option('--activation', default='relu,relu,linear', help='Comma-separated activation functions')
@click.option('--loss', default='cross_entropy', help='Loss function: cross_entropy, mse_loss, nll_loss')
@click.option('--optimizer', default='adam', help='Optimizer: adam, sgd')
@click.option('--lr', default=0.001, type=float, help='Learning rate')
@click.option('--epochs', default=5, type=int, help='Number of training epochs')
@click.option('--batch-size', default=32, type=int, help='Batch size for training')
@click.option('--save-model', default=None, help='Path to save trained model')

def train(layers: str, activation: str, loss: str, optimizer: str, lr: float, 
          epochs: int, batch_size: int, save_model: str | None) -> None:
    """Train a neural network on MNIST dataset."""
    click.echo(f"Training neural network with layers: {layers}")
    
    #Parse layer sizes and activation functions
    layer_sizes: list[int] = list(map(int, layers.split(',')))
    activations: list[str] = activation.split(',')
    
    #Validate input lengths
    if len(activations) != len(layer_sizes) - 1:
        raise ValueError("Number of activation functions must be one less than number of layers.")
    
    #Create model configuration
    config: dict = {
        "loss": loss,
        "optimizer": optimizer,
        "lr": lr,
        "activations": activations,
    }
    
    # Create model - passes layer_sizes and config dict
    model: Network = Network(layer_sizes=layer_sizes, config=config)
    
