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