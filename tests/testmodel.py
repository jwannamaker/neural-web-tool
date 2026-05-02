#!/usr/bin/env python
import pytest
import torch
from neuralwebtool.model import Network

class TestNetwork:
    def test_init(self):
        """Test network init with default parameters"""
        model = Network()
        assert len(model.layers) == 3 
        assert len(model.activations) == 3
        assert 'input' in model.layers
        assert 'hidden_0' in model.layers
        assert 'output' in model.layers

    def test_init_custom_layers(self):
        """Test network init with custom layer sizes and activations"""
        layer_sizes = [10,20,5]
        config = {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
            "activations": ["relu", "sigmoid"],
        }
        model = Network(layer_sizes, config)
        assert len(model.layers) == 2
        assert model.layers['input'].in_features == 10
        assert model.layers['input'].out_features == 20
        assert model.layers['output'].in_features == 20
        assert model.layers['output'].out_features == 5

    def test_init_invalid_activations_length(self):
        """Test that ValueError is raised for mismatched activations length"""
        layer_size = [10,20,5]
        config = {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
            "activations": ["relu"]
        }
        with pytest.raises(ValueError, match="number of activation function"):
            Network(layer_sizes, config)

    def test_forward(self):
        """Test if forward pass produces output of correct shape"""
        layer_sizes = [10,20,5]
        config = {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
            "activations": ["relu", "sigmoid"],
        }
        model = Network(layer_sizes, config)
        input_data = torch.randn(4,10) #batch of 4 samples, each with 10 features
        output = model(input_data)
        assert output.shape == (4,5) #output should have shape (batch_size, output_features)

    