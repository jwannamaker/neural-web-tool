import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reLU(x):
    if x > 0:
        return x
    return 0


def sigmoid_prime(x):
    """Derivative of sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


class Network(object):
    def __init__(self, sizes: list[int]):
        """
        nnet = Network([2, 3, 1])
        Would create a network with 2 neurons in the first layer, 3 neurons in the second layer, and 1 neuron in the final layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Returns output if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(
        self, training_data, epochs, mini_batch_size, eta, test_data=None
    ):
        """
        Args:
            training_data (list[tuple[2]]): list of tuples (x,y) representing the training inputs and corresponding desired outputs
            epochs (int): The number of epochs to train for
            eta: learning rate
            test_data: What to evaluate against after each epoch of training.
        """
        from random import shuffle

        n_test = len(test_data) if test_data else 0
        n = len(training_data)

        for j in range(epochs):
            shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropogation to a single mini batch."""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]

        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, grad_b)
        ]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, grad_w)
        ]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        a = x
        activations = [x]  # store all the activations layer by layer
        z = 0
        z_list = []  # store z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            z_list.append(z)
            a = sigmoid(z_list[-1])  # might cause an error later, comeback
            activations.append(a)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_list[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = z_list[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (grad_b, grad_w)

    def evaluate(self, test_data):
        """Return number of test inputs the network outputs correct result for."""
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return output_activations - y


def main():
    training_data =  
    nn = Network([16, 8, 8, 4])
    nn.stochastic_gradient_descent(training_data, 
