"""Integration tests for the complete neural network system."""
import numpy as np
from neural_network import NeuralNetwork


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_training_pipeline(self):
        """Test complete training and prediction pipeline."""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)

        x_train = [np.array([0.1, 0.1]), np.array([0.9, 0.9])]
        y_train = [np.array([0.1]), np.array([0.9])]

        initial_outputs = [network.forward(x) for x in x_train]

        for _ in range(50):
            for inputs, targets in zip(x_train, y_train):
                network.train(inputs, targets, learning_rate=0.5)

        final_outputs = [network.forward(x) for x in x_train]

        initial_error = sum(abs(initial_outputs[i][0] - y_train[i][0])
                            for i in range(len(x_train)))
        final_error = sum(abs(final_outputs[i][0] - y_train[i][0])
                          for i in range(len(x_train)))

        assert final_error <= initial_error, \
            "Network should reduce error with training"

    def test_different_architectures(self):
        """Test network with various layer configurations."""
        architectures = [
            [1, 1],
            [2, 2],
            [3, 2, 1],
            [4, 8, 4, 1],
            [5, 10, 5, 2],
        ]

        for layer_sizes in architectures:
            network = NeuralNetwork(layer_sizes)
            inputs = np.random.rand(layer_sizes[0])
            outputs = network.forward(inputs)

            assert outputs.shape == (layer_sizes[-1],), \
                f"Network {layer_sizes} should output {layer_sizes[-1]}"
