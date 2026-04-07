import numpy as np          

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)  #vector of weights for each input
        self.bias = np.random.randn()               #single bias
        self.input = None                           #vector
        self.output = None                          #scalar
    
    def activate(self, x):
        #x is the vector of input data
        self.input = x
        self.input_activate = np.dot(self.weights, x) + self.bias
        self.output = 1 / (1 + np.exp(-self.input_activate))  #Sigmoid function to normalize the data
        return self.output
    
    def forward(self, input_vector):
        return self.activate(input_vector)
    
    def back_propagate(self, target, learning_rate):
        #error = output - target
        error = self.output - target
        #derivative of sigmoid function
        delta = error * self.output * (1 - self.output)
        #udpate weights and bias
        self.weights -= learning_rate * delta * self.input
        self.bias -= learning_rate * delta
        return delta  #return delta to use for backpropagation in prev layers

#layer class holds multiple neurons and manages the forward and backward passes
class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        #inputs is a vector of inputs that are inputted to the layer
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return np.array(outputs)
    
    def back_propagate(self, deltas, learning_rate):
        #deltas is a vector of deltas from next layer
        for i, neuron in enumerate(self.neurons):
            neuron.back_propagate(deltas[i], learning_rate)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        #layer_sizes: list like [input_size, hidden1_size, ..., output_size]
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1]))
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs  #final output
    
    def train(self, inputs, targets, learning_rate):
        #Forward pass
        outputs = self.forward(inputs)
        #Backpropagate from output layer
        deltas = outputs - targets  # targets are the expected outputs
        for layer in reversed(self.layers):
            layer.back_propagate(deltas, learning_rate)
            #For hidden layers, deltas would be computed from next layer's weights, need to implement that logic 
            #requires access to weights of next layer, which needs to be implemented.
        
