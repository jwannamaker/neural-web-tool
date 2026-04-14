"""Neural network implementation with neuron, layer, and network classes."""
import numpy as np


class Neuron:
    """A single neuron with weights, bias, and sigmoid activation function."""

    def __init__(self, num_inputs):
        """Initialize neuron with random weights and bias.

        Args:
            num_inputs: Number of input connections to this neuron.
        """
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.input = None
        self.output = None
        self.input_activate = None

    def activate(self, x):
        """Apply sigmoid activation function to weighted input plus bias.

        Args:
            x: Input vector to the neuron.

        Returns:
            Sigmoid activation output (between 0 and 1).
        """
        self.input = x
        self.input_activate = np.dot(self.weights, x) + self.bias
        self.output = 1 / (1 + np.exp(-self.input_activate))
        return self.output

    def forward(self, input_vector):
        """Forward pass through the neuron.

        Args:
            input_vector: Input to the neuron.

        Returns:
            Output from sigmoid activation function.
        """
        return self.activate(input_vector)

    def back_propagate(self, target, learning_rate):
        """Backpropagation step to update weights and bias.

        Args:
            target: Target output value.
            learning_rate: Learning rate for weight updates.

        Returns:
            Delta value for backpropagation to previous layers.
        """
        error = self.output - target
        delta = error * self.output * (1 - self.output)
        self.weights -= learning_rate * delta * self.input
        self.bias -= learning_rate * delta
        return delta


class Layer:
    """A layer of neurons that processes inputs in parallel.

    Attributes:
        neurons: List of neuron objects in this layer.
    """

    def __init__(self, num_neurons, num_inputs_per_neuron):
        """Initialize a layer with multiple neurons.

        Args:
            num_neurons: Number of neurons in this layer.
            num_inputs_per_neuron: Number of inputs each neuron receives.
        """
        self.neurons = [Neuron(num_inputs_per_neuron)
                        for _ in range(num_neurons)]

    def forward(self, inputs):
        """Forward pass through all neurons in the layer.

        Args:
            inputs: Input vector to the layer.

        Returns:
            Array of outputs from all neurons.
        """
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return np.array(outputs)

    def back_propagate(self, deltas, learning_rate):
        """Backpropagation through all neurons in the layer.

        Args:
            deltas: Delta values from the next layer.
            learning_rate: Learning rate for weight updates.
        """
        for i, neuron in enumerate(self.neurons):
            neuron.back_propagate(deltas[i], learning_rate)


class NeuralNetwork:
    """A multi-layer neural network with forward and backward propagation.

    Attributes:
        layers: List of Layer objects.
    """

    def __init__(self, layer_sizes):
        """Initialize neural network with specified layer architecture.

        Args:
            layer_sizes: List of layer sizes, e.g.,
                [input_size, hidden_size, output_size]
        """
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1]))

    def forward(self, inputs):
        """Forward pass through all layers.

        Args:
            inputs: Input vector to the network.

        Returns:
            Output from the final layer.
        """
        self.layer_outputs = [inputs]
        for layer in self.layers:
            inputs = layer.forward(inputs)
            self.layer_outputs.append(inputs)
        return inputs

    def train(self, inputs, targets, learning_rate):
        """Training step with forward and backward propagation.

        Args:
            inputs: Input vector.
            targets: Target output values.
            learning_rate: Learning rate for weight updates.
        """
        outputs = self.forward(inputs)

        #data for the output layer
        deltas = []
        for i, output in enumerate(outputs):
            error = output - targets[i]
            delta = error * output * (1 - output)
            deltas.append(delta)
        deltas = np.array(deltas)

        #going in reverse order through the layers to backpropagate
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            new_deltas = []

            for neuron_idx, neuron in enumerate(layer.neurons):
                if layer_idx == len(self.layers) - 1:
                    #using output layer deltas
                    delta = deltas[neuron_idx]
                else:
                    #compute deltas for the hidden layers
                    next_layer = self.layers[layer_idx + 1]
                    weighted_sum = 0
                    for next_neuron_idx, next_neuron in \
                            enumerate(next_layer.neurons):
                        weighted_sum += \
                            next_neuron.weights[neuron_idx] * \
                            deltas[next_neuron_idx]
                    delta = weighted_sum * neuron.output * \
                        (1 - neuron.output)
                    new_deltas.append(delta)

                #update weights and bias for the current neuron we are at
                neuron.weights -= learning_rate * delta * \
                    self.layer_outputs[layer_idx]
                neuron.bias -= learning_rate * delta

            if layer_idx > 0:
                deltas = np.array(new_deltas)

    def predict(self, inputs):
        """Make predictions using the trained network.

        Args:
            inputs: Input vector.

        Returns:
            Predicted output from the network.
        """
        return self.forward(inputs)

    def compute_loss(self, input, targets):
        """Compute mean squared error loss for given input and targets.

        Args:
            input: Input vector.
            targets: Target output values.
        Returns:
            Mean squared error loss.
        """
        outputs = self.forward(input)
        #the mean of (y_hat - y)^2 for all output neurons
        loss = np.mean((outputs - targets) ** 2)
        return loss




