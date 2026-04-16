"""Single neuron implementation with sigmoid activation."""
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




