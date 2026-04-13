import pytest
import nuimport numpy as np
from Neuron import Neuron, Layer, NeuralNetwork
class TestNeuron:
    
    def test_neuron_initialization(self):
        """Test that the neuron initializes with correct weight and bias"""
        num_inputs = 3
        neuron = Neuron(num_inputs)
        
        assert neuron.weights.shape == (num_inputs,), "Weights shape should match num_inputs"
        assert isinstance(neuron.bias, (float, np.floating)), "Bias should be a float"
        assert neuron.input is None, "Input should start as None"
        assert neuron.output is None, "Output should start as None"
    




