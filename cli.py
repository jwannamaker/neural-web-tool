#!/usr/bin/env python
"""Command-line interface for neural network training and experimentation."""
import click
import torch

from neuralwebtool.model import Network
from neuralwebtool.trainer import Trainer
from neuralwebtool.data import Data


@click.group(
    help="Neural Web Tool CLI. Use subcommands to create, train, or evaluate neural networks."
)
def cli() -> None:
    """Neural Web Tool - Train and test neural networks from the command line."""
    pass


@cli.command(help='Train a neural network on MNIST using configurable layers, activations, optimizer and loss. Use --save-model to save the trained model weights to disk.')
@click.option('--layers', default='784,128,64,10', help='Comma-separated layer sizes, including input and output. Example: 784,256,128,10')
@click.option('--activation', default='relu,relu,linear', help='Comma-separated activation functions, one per layer transition. Supported: relu, sigmoid, tanh, linear')
@click.option('--loss', default='cross_entropy', help='Loss function to train with. Supported: cross_entropy, mse_loss, nll_loss')
@click.option('--optimizer', default='adam', help='Optimizer to use. Supported: adam, sgd')
@click.option('--lr', default=0.001, type=float, help='Learning rate for training updates')
@click.option('--epochs', default=5, type=int, help='Number of training epochs')
@click.option('--batch-size', default=32, type=int, help='Batch size for training')
@click.option('--save-model', default=None, help='Optional path to save trained model weights (e.g. model.pt). The model will be saved after training completes.')

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
    
    #Create model: passes layer_sizes and config dict
    model: Network = Network(layer_sizes=layer_sizes, config=config)
    #Load data
    click.echo("Loading MNIST dataset...")
    data: Data = Data()
    train_loader = data.get_dataloader(batch_size=batch_size, train=True)
    
    #Create trainer with model config
    trainer: Trainer = Trainer(model, config)
    
    #Training loop
    for epoch in range(epochs):
        batch_count: int = 0
        
        with click.progressbar(train_loader, label=f'Epoch {epoch+1}/{epochs}') as bar:
            for images, labels in bar:
                #Flatten images for fully connected network
                images = images.reshape(images.size(0), -1)
                trainer.train_step(images, labels)
                batch_count += 1
        
        click.echo(f"Epoch (Training cycle) {epoch+1} completed")
    
    # Save model if requested
    if save_model:
        torch.save(model.state_dict(), save_model)
        click.echo(f"Model saved to {save_model}")
    
    click.echo("Training complete!")

@cli.command(help='Evaluate a saved model on the MNIST test dataset.')
@click.option('--model-path', required=True, help='Path to saved model weights file')
@click.option('--layers', default='784,128,64,10', help='Comma-separated layer sizes matching the saved model architecture')
@click.option('--batch-size', default=32, type=int, help='Batch size for evaluation')
def evaluate(model_path: str, layers: str, batch_size: int) -> None:
    """Evaluate a trained model on MNIST test dataset."""
    click.echo(f"Loading model from {model_path}")
    
    layer_sizes: list[int] = list(map(int, layers.split(',')))
    config: dict = {
        "loss": "cross_entropy",
        "optimizer": "adam",
        "lr": 0.001,
        "activations": ["relu", "relu", "linear"],
    }
    
    #Load model
    model: Network = Network(layer_sizes=layer_sizes, config=config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    #Load test data
    click.echo("Loading test dataset...")
    data: Data = Data()
    test_loader = data.get_dataloader(batch_size=batch_size, train=False)
    #Evaluate
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy: float = 100 * correct / total
    click.echo(f"Accuracy: {accuracy:.2f}%")

@cli.command(help='Create and display a network architecture without training.')
@click.option('--layers', default='784,128,64,10', help='Comma-separated layer sizes, including input and output. Example: 784,256,128,10')
def create_network(layers: str) -> None:
    """Create and display a network architecture."""
    layer_sizes: list[int] = list(map(int, layers.split(',')))
    
    config: dict = {
        "loss": "cross_entropy",
        "optimizer": "adam",
        "lr": 0.001,
        "activations": ["relu", "relu", "linear"],
    }
    
    model: Network = Network(layer_sizes=layer_sizes, config=config)
    click.echo("Network Architecture:")
    click.echo(model)

if __name__ == '__main__':
    cli()




