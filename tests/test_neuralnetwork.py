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
    
    def test_neuron_forward_pass(self):
        """Test that neuron forward pass produces output between 0 and 1 (verification that sigmoid is correct)"""
        neuron = Neuron(2)
        inputs = np.array([0.5, 0.8])
        
        output = neuron.forward(inputs)
        
        #checks output
        assert 0 <= output <= 1, "Sigmoid output should be between 0 and 1"
        #checks if input is stored
        assert np.allclose(neuron.input, inputs), "Input should be stored"
        #checks if output is stored
        assert neuron.output == output, "Output should be stored"

      def test_neuron_activate_same_as_forward(self):
        """Test that activate and forward produce the same result"""
        neuron = Neuron(2)
        inputs = np.array([0.3, 0.7])
        
        output1 = neuron.activate(inputs)
        neuron2 = Neuron(2)
        neuron2.weights = neuron.weights  #same weights
        neuron2.bias = neuron.bias
        output2 = neuron2.forward(inputs)
        
        assert np.allclose(output1, output2), "activate and forward should give same result"
    



