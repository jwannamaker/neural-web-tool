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
    
    def test_neuron_back_propagation_updates_weights(self):
        """Test that backpropagation updates weights and bias"""
        neuron = Neuron(2)
        inputs = np.array([0.5, 0.8])
        target = 1.0
        learning_rate = 0.1
        
        #store og weights
        original_weights = neuron.weights.copy()
        original_bias = neuron.bias
        
        #forward oass
        neuron.forward(inputs)
        
        #backwards
        delta = neuron.back_propagate(target, learning_rate)
        
        #checks if weights and biases were changed
        assert not np.allclose(neuron.weights, original_weights), "Weights should be updated"
        assert neuron.bias != original_bias, "Bias should be updated"
        #checks if delta is returned
        assert isinstance(delta, (float, np.floating)), "Delta should be a float"
    
  def test_neuron_sigmoid_properties(self):
        """Test sigmoid function properties"""
        neuron = Neuron(1)
        
        #test with zero
        neuron.weights = np.array([0.0])
        neuron.bias = 0.0
        output = neuron.forward(np.array([0.0]))
        assert np.isclose(output, 0.5), "Sigmoid(0) should be 0.5"
        
        #test with very large positive
        neuron.weights = np.array([1.0])
        neuron.bias = 0.0
        output = neuron.forward(np.array([10.0]))
        assert output > 0.99, "Sigmoid of large positive should be close to 1"

class TestLayer:
    """Test cases for the Layer class"""
    
    def test_layer_initialization(self):
        """Test that layer initializes with correct number of neurons"""
        num_neurons = 4
        num_inputs = 3
        layer = Layer(num_neurons, num_inputs)
        
        assert len(layer.neurons) == num_neurons, "Should have correct number of neurons"
        for neuron in layer.neurons:
            assert neuron.weights.shape == (num_inputs,), "Each neuron should have correct input size"
    
    def test_layer_forward_pass_shape(self):
        """Test that layer forward pass returns correct shape"""
        num_neurons = 4
        num_inputs = 3
        layer = Layer(num_neurons, num_inputs)
        
        inputs = np.array([0.5, 0.8, 0.3])
        outputs = layer.forward(inputs)
        
        assert outputs.shape == (num_neurons,), "Output shape should match number of neurons"
        assert all(0 <= o <= 1 for o in outputs), "All outputs should be between 0 and 1"
    
    def test_layer_back_propagation(self):
        """Test that layer backpropagation updates all neurons"""
        num_neurons = 3
        num_inputs = 2
        layer = Layer(num_neurons, num_inputs)
        
        
        inputs = np.array([0.5, 0.8])
        outputs = layer.forward(inputs)
        
        
        original_weights = [neuron.weights.copy() for neuron in layer.neurons]
        
        
        deltas = np.array([0.1, 0.2, 0.15])
        layer.back_propagate(deltas, learning_rate=0.1)
        
        #check if all weights were updated
        for i, neuron in enumerate(layer.neurons):
            assert not np.allclose(neuron.weights, original_weights[i]), \
                f"Neuron {i} weights should be updated"


class TestNeuralNetwork:
    """Test cases for the NeuralNetwork class"""
    
    def test_network_initialization(self):
        """Test that network initializes with correct number of layers"""
        layer_sizes = [2, 4, 3, 1]
        network = NeuralNetwork(layer_sizes)
        
        #there should be three layers
        expected_layers = len(layer_sizes) - 1
        assert len(network.layers) == expected_layers, "Should have correct number of layers"
        
        #check layer sizes
        assert len(network.layers[0].neurons) == 4, "First hidden layer should have 4 neurons"
        assert len(network.layers[1].neurons) == 3, "Second hidden layer should have 3 neurons"
        assert len(network.layers[2].neurons) == 1, "Output layer should have 1 neuron"
    
    def test_network_forward_pass_shape(self):
        """Test that network forward pass returns correct shape"""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)
        
        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)
        
        #output shape should match output layer size
        expected_size = layer_sizes[-1]
        assert isinstance(outputs, np.ndarray), "Output should be numpy array"
        assert outputs.shape == (expected_size,), f"Output shape should be ({expected_size},)"
    
    def test_layer_forward_pass_shape(self):
        """Test that layer forward pass returns correct shape"""
        num_neurons = 4
        num_inputs = 3
        layer = Layer(num_neurons, num_inputs)
        
        inputs = np.array([0.5, 0.8, 0.3])
        outputs = layer.forward(inputs)
        
        assert outputs.shape == (num_neurons,), "Output shape should match number of neurons"
        assert all(0 <= o <= 1 for o in outputs), "All outputs should be between 0 and 1"
    
    def test_layer_back_propagation(self):
        """Test that layer backpropagation updates all neurons"""
        num_neurons = 3
        num_inputs = 2
        layer = Layer(num_neurons, num_inputs)
        
        
        inputs = np.array([0.5, 0.8])
        outputs = layer.forward(inputs)
        
        
        original_weights = [neuron.weights.copy() for neuron in layer.neurons]
        
        
        deltas = np.array([0.1, 0.2, 0.15])
        layer.back_propagate(deltas, learning_rate=0.1)
        
        #check if all weights were updated
        for i, neuron in enumerate(layer.neurons):
            assert not np.allclose(neuron.weights, original_weights[i]), \
                f"Neuron {i} weights should be updated"


class TestNeuralNetwork:
    """Test cases for the NeuralNetwork class"""
    
    def test_network_initialization(self):
        """Test that network initializes with correct number of layers"""
        layer_sizes = [2, 4, 3, 1]
        network = NeuralNetwork(layer_sizes)
        
        #there should be three layers
        expected_layers = len(layer_sizes) - 1
        assert len(network.layers) == expected_layers, "Should have correct number of layers"
        
        #check layer sizes
        assert len(network.layers[0].neurons) == 4, "First hidden layer should have 4 neurons"
        assert len(network.layers[1].neurons) == 3, "Second hidden layer should have 3 neurons"
        assert len(network.layers[2].neurons) == 1, "Output layer should have 1 neuron"
    
    def test_network_forward_pass_shape(self):
        """Test that network forward pass returns correct shape"""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)
        
        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)
        
        # Output shape should match output layer size
        expected_size = layer_sizes[-1]
        assert isinstance(outputs, np.ndarray), "Output should be numpy array"
        assert outputs.shape == (expected_size,), f"Output shape should be ({expected_size},)"
    
    def test_network_forward_pass_range(self):
        """Test that network outputs are in valid range (0-1 due to sigmoid)"""
        layer_sizes = [2, 4, 2]
        network = NeuralNetwork(layer_sizes)
        
        inputs = np.array([0.5, 0.8])
        outputs = network.forward(inputs)
        
        assert all(0 <= o <= 1 for o in outputs), "All outputs should be between 0 and 1"
    
    def test_network_training_updates_weights(self):
        """Test that training updates network weights"""
        layer_sizes = [2, 3, 1]
        network = NeuralNetwork(layer_sizes)
        
        #store original weights
        original_weights = []
        for layer in network.layers:
            for neuron in layer.neurons:
                original_weights.append(neuron.weights.copy())
        
        #training
        inputs = np.array([0.5, 0.8])
        targets = np.array([1.0])
        network.train(inputs, targets, learning_rate=0.1)
        
        #check that at least some weights changed
        weights_changed = False
        for i, layer in enumerate(network.layers):
            for j, neuron in enumerate(layer.neurons):
                if not np.allclose(neuron.weights, original_weights[i * len(original_weights) // len(network.layers) + j]):
                    weights_changed = True
                    break
        
        assert weights_changed, "Training should update at least some weights"
    
  





