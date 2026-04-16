"""Unit tests for the Layer class."""
import numpy as np
from layer import Layer


class TestLayer:
    """Test cases for the Layer class."""

    def test_layer_initialization(self):
        """Test that layer initializes with correct number of neurons."""
        num_neurons = 4
        num_inputs = 3
        layer = Layer(num_neurons, num_inputs)

        assert len(layer.neurons) == num_neurons, \
            "Should have correct number of neurons"
        for neuron in layer.neurons:
            assert neuron.weights.shape == (num_inputs,), \
                "Each neuron should have correct input size"

    def test_layer_forward_pass_shape(self):
        """Test that layer forward pass returns correct shape."""
        num_neurons = 4
        num_inputs = 3
        layer = Layer(num_neurons, num_inputs)

        inputs = np.array([0.5, 0.8, 0.3])
        outputs = layer.forward(inputs)

        assert outputs.shape == (num_neurons,), \
            "Output shape should match number of neurons"
        assert all(0 <= o <= 1 for o in outputs), \
            "All outputs should be between 0 and 1"

    def test_layer_back_propagation(self):
        """Test that layer backpropagation updates all neurons."""
        num_neurons = 3
        num_inputs = 2
        layer = Layer(num_neurons, num_inputs)

        inputs = np.array([0.5, 0.8])
        layer.forward(inputs)

        original_weights = [neuron.weights.copy() for neuron in layer.neurons]

        deltas = np.array([0.1, 0.2, 0.15])
        layer.back_propagate(deltas, learning_rate=0.1)

        for i, neuron in enumerate(layer.neurons):
            assert not np.allclose(neuron.weights, original_weights[i]), \
                f"Neuron {i} weights should be updated"
