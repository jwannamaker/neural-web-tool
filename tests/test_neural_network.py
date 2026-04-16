"""Unit tests for the NeuralNetwork class."""
import numpy as np
from neural_network import NeuralNetwork


class TestNeuralNetwork:
    """Test cases for the NeuralNetwork class."""

    def test_network_initialization(self):
        """Test that network initializes with correct number of layers."""
        layer_sizes = [2, 4, 3, 1]
        network = NeuralNetwork(layer_sizes)

        expected_layers = len(layer_sizes) - 1
        assert len(network.layers) == expected_layers, \
            "Should have correct number of layers"

        assert len(network.layers[0].neurons) == 4, \
            "First hidden layer should have 4 neurons"
        assert len(network.layers[1].neurons) == 3, \
            "Second hidden layer should have 3 neurons"
        assert len(network.layers[2].neurons) == 1, \
            "Output layer should have 1 neuron"

    def test_network_forward_pass_shape(self):
        """Test that network forward pass returns correct shape."""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)

        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)

        expected_size = layer_sizes[-1]
        assert isinstance(outputs, np.ndarray), "Output should be numpy array"
        assert outputs.shape == (expected_size,), \
            f"Output shape should be ({expected_size},)"

    def test_network_forward_pass_range(self):
        """Test that network outputs are valid (0-1 due to sigmoid)."""
        layer_sizes = [2, 4, 2]
        network = NeuralNetwork(layer_sizes)

        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)

        assert all(0 <= o <= 1 for o in outputs), \
            "All outputs should be between 0 and 1"

    def test_network_training_updates_weights(self):
        """Test that training updates network weights."""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)

        original_weights = []
        for layer in network.layers:
            for neuron in layer.neurons:
                original_weights.append(neuron.weights.copy())

        inputs = np.array([0.5, 0.8])
        targets = np.array([1.0])
        network.train(inputs, targets, learning_rate=0.1)

        weights_changed = False
        idx = 0
        for layer in network.layers:
            for neuron in layer.neurons:
                if idx < len(original_weights) and \
                   not np.allclose(neuron.weights, original_weights[idx]):
                    weights_changed = True
                    break
                idx += 1

        assert weights_changed, "Training should update at least some weights"

    def test_network_multiple_outputs(self):
        """Test network with multiple output neurons."""
        layer_sizes = [2, 3, 3]
        network = NeuralNetwork(layer_sizes)

        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)

        assert outputs.shape == (3,), "Should have 3 outputs"
        assert all(0 <= o <= 1 for o in outputs), \
            "All outputs should be valid"

    def test_network_with_single_neuron(self):
        """Test network with minimal architecture."""
        layer_sizes = [1, 1]
        network = NeuralNetwork(layer_sizes)

        inputs = np.array([0.5])
        outputs = network.forward(inputs)

        assert outputs.shape == (1,), "Should have 1 output"
        assert 0 <= outputs[0] <= 1, "Output should be valid"

        network.train(inputs, np.array([1.0]), learning_rate=0.1)
