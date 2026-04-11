# Reading "Neural Networks and Deep Learning"

## Chapter 1: Exercises
1. Sigmoid neurons simulating perceptrons

    - Take all the weights and biases in a network of perceptrons, and multiply them by some positive constant c > 0. Show that the behavior of the network doesn't change.

    - If the sigmoid neuron is ``` 1 / (1 + e^-z)``` where z is ```w*x + b``` and x is all the inputs to a neuron, then multiplying the same constant to each of the weights and biases would give us ```cw*x + cb``` which is ```c(w*x + b)``` as the output of each sigmoid neuron. Furthermore, when differentiating instead of getting 
```
delta_output \approx \sum_j \frac{del_output}{del_weights}*
```

---
# Watching Neural Networks by 3blue1brown

## Gradient descent
- Gradient Descent is how the network "learns"

- A Neuron is more like a function than like something that "holds" a number
* Input:        All the activations of the previous layer
* Output:       A number [0, 1]
* Parameters:   Weights and biases of each neuron in the previous layer

- Neural Network is basically like a function
* Input:        784 pixel numbers [0, 255] "whiteness"
* Output:       10 numbers, one of them being the highest and closest to 1
* Parameters:   13,002 weights and biases

- Cost function
* Input:        13,002 weights and biases
* Output:       1 number, the cost
* Parameters:   10,000s of training examples

- Explaining the cost function
- Take any point on a function, and step in the direction where the output gets lower
- Local minimum is doable, but absolute minimum is extremely difficult
- Which direction should you step in the input space to decrease the output of the cost function the most quickly?
- What's the "downhill" direction.
- The gradient give the direction of steepest ascent
- The negative of the gradient will naturally give us the steepest downhill descent
- It tell you which nudges to all of the weights and biases will create the quickest decrease in the cost output


## How is Neural Network like a matrix operation?
- Next layer's activations = sigmoid(dot_product(weights, activations) + biases)

```
Apply activation    weights of each neuron             activations of     biases of each
function to each    in the previous layer              each neuron in     neuron in the
component of the    corresponding to each              previous layer     next layer
resulting vector    neuron in the next layer

                    |  w[0,0]  w[0,1]  w[0,2]  |       |  a[0]  |          |  b[0]  |
                    |  w[1,0]  w[1,1]  w[1,2]  |       |  a[1]  |          |  b[1]  |
                    |  w[2,0]  w[2,1]  w[2,2]  |   *   |  a[2]  |     +    |  b[2]  |
                    |  w[3,0]  w[3,1]  w[3,2]  |       |  a[3]  |          |  b[3]  |
                    |  w[4,0]  w[4,1]  w[4,2]  |       |  a[4]  |          |  b[4]  |
```

- Some Python corresponding to this
```
class Network(object):
    def __init__(self, *args, **kwargs):
        # set weights and biases

    def feed_forward(self, a):
        """ Return the output of the network for an input vector a """
        for b, w in zipo(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
```

## Cost function
- Uses a test case (with known desired output) to find the mean squared difference 
- average( SUM[ (Actual activations - Expected activations)^2 ] )
- Find the derivative of the cost function and 
    - If slope is positive shift (weight/bias) to the left
    - If slope is negative shift (weight/bias) to the right
- Can be interpreted as the answer to the question: Which weights matter the most to get the desired output?

## Backpropagation





---
# Tech stack to possibly use for this project
- Three.js
- TensorFlow.js
- WebGL



