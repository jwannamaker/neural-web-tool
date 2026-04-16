"""Unit tests for the Neuron class."""
import numpy as np
from Neuron import Neuron


class TestNeuron:
    """Test cases for the Neuron class."""

    def test_neuron_initialization(self):
        """Test that the neuron initializes with correct weight and bias."""
        num_inputs = 3
        neuron = Neuron(num_inputs)

        assert neuron.weights.shape == (num_inputs,), \
            "Weights shape should match num_inputs"
        assert isinstance(neuron.bias, (float, np.floating)), \
            "Bias should be a float"
        assert neuron.input is None, "Input should start as None"
        assert neuron.output is None, "Output should start as None"

    def test_neuron_forward_pass(self):
        """Test that neuron forward pass produces output between 0 and 1."""
        neuron = Neuron(2)
        inputs = np.array([0.5, 0.8])

        output = neuron.forward(inputs)

        assert 0 <= output <= 1, "Sigmoid output should be between 0 and 1"
        assert np.allclose(neuron.input, inputs), "Input should be stored"
        assert neuron.output == output, "Output should be stored"

    def test_neuron_activate_same_as_forward(self):
        """Test that activate and forward produce the same result."""
        neuron = Neuron(2)
        inputs = np.array([0.3, 0.7])

        output1 = neuron.activate(inputs)
        neuron2 = Neuron(2)
        neuron2.weights = neuron.weights
        neuron2.bias = neuron.bias
        output2 = neuron2.forward(inputs)

        assert np.allclose(output1, output2), \
            "activate and forward should give same result"

    def test_neuron_back_propagation_updates_weights(self):
        """Test that backpropagation updates weights and bias."""
        neuron = Neuron(2)
        inputs = np.array([0.5, 0.8])
        target = 1.0
        learning_rate = 0.1

        original_weights = neuron.weights.copy()
        original_bias = neuron.bias

        neuron.forward(inputs)

        delta = neuron.back_propagate(target, learning_rate)

        assert not np.allclose(neuron.weights, original_weights), \
            "Weights should be updated"
        assert neuron.bias != original_bias, "Bias should be updated"
        assert isinstance(delta, (float, np.floating)), \
            "Delta should be a float"

    def test_neuron_sigmoid_properties(self):
        """Test sigmoid function properties."""
        neuron = Neuron(1)

        neuron.weights = np.array([0.0])
        neuron.bias = 0.0
        output = neuron.forward(np.array([0.0]))
        assert np.isclose(output, 0.5), "Sigmoid(0) should be 0.5"

        neuron.weights = np.array([1.0])
        neuron.bias = 0.0
        output = neuron.forward(np.array([10.0]))
        assert output > 0.99, "Sigmoid of large positive should be close to 1"
