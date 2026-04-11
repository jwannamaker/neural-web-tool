import pytest

class TestNeuralNetwork:
    def test_REQ_001_nn_created_with_layer_sizes(self):
        from neuralnetwork import NeuralNetwork
        nn = NeuralNetwork(input_size = 784, hidden_size = 16, output_size = 10)
        assert nn.input_size == 28 * 28

