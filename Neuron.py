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

    # Activation Function - the "processing" part of the neuron, takes the input and output
    # and normalizes the data to be between 0 and 1, which allows for 
    # complex patterns to be learned.
    # Algorithm that allows for non-linear patterns
    # def avtivation(input_activate):
    # Sigmoid function, gives output between 1 and 0 which "squeezes" data to be between 1 and 0
    # which allows for more complex, non-linear patterns to be learned
    def activate(self, x):
        self.input_activate = (self.input * self.weight) + self.bias 
        self.output = 1 / (1 + np.exp(-self.input_activate))  
        return self.output
    

    #Back propagation and forward function are used together, forward function 
    #is used to make a prediction, then back propagation is used to adjust
    #weights and bias to minimize error. this is how the neuron learns

    # Back Propagation function - essentially the "learning" part of the neuron
    # Algorithm that calculated error with respect to weight to minimize error
    # def back_propagation(output)
    #uses the derivative of the activation (sigmoid) function to calculate error with respect to weight

    #Back propagation has two parts, a forward pass and a backward pass.
    # the forward pass calculates the output of the neuron, then the backward pass calculates the error
    # with respect to the values and adjusts to minimize error. 
    
    def back_propagation(self, target, learning_rate):
        error = self.putput - target #calculate error
        delta = error * self.output * (1 - self.output) #calculate delta using derivative of sigmoid function
        #change the weight and bias
        self.weight -= learning_rate * delta * self.last_input
        self.bias -= learning_rate * delta

    #Forward function - essentially the prediction part of the neuron
    #Explanation: Input recieves raw data then normalizes the data
    # activate function takes the normalized data and applies the values, then gives the output to 
    # the next layer or final prediction
    def forward(self, input):
        self.input = input
        return self.activate(input)
        
        
