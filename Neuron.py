import numpy as np            # used to import numpy arrays to store data to be processed

Class Neuron:
    weight = 0.00             # Float - how much "influence" a neuron has before activation function
    bias = 0.00               # Float - adjust output of neuron after weights are calculated, allows fine-tuning
    input = 0.00              # Float - gets input from data or output from previous neuron layer
    input_activate = 0.00     # Float - (input times weight) plus bias, result given to activation function
    output = 0.00             # Float = final output from neuron to next layer or final prediction after activation function
    
   
    # Parameter Constructor
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.input = self
        self.input_activate = self
        self.output = self

    # Activation Function
    # Algorithm that allows for non-linear patterns
    # def avtivation(input_activate):

    # Back Propagation function
    # Algorithm that calculated error with respect to weight to minimize error
    # def back_propagation(output)
        