"""Neural network layer implementation."""
import numpy as np
from Neuron import Neuron


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
