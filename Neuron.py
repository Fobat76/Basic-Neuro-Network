#BASIC EXAMPLE OF A NEURO NETWORK WITHOUT TRAINING Additonally every neuron should have it's own weight and bias. optimal for changing the network up for optimal accuarcy
#I am unable to design the network to take in certain data sets and optimize the network. As my linear algebra(Up to Matrix Algebra) and Calculus skills are limited (Up to Calculus 2)
#I hope to improve this design in the future as I learn more


import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x)) 
#Sigmoid function 

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias
    # Here is our Neuron constructor 
  def feedforward(self, inputs):
    total = np.dot(self.weights, inputs) + self.bias #Caculate the weighted sum 
    return sigmoid(total)   #Return the result 

class Neuron_Network:
    def __init__(self):
        weights = 3
        bias = 2

        self.h1n1 = Neuron(weights, bias)   #First Hidden Layer
        self.h1n2 = Neuron(weights, bias)
        self.h1n3 = Neuron(weights, bias)
        self.h1n4 = Neuron(weights, bias)

        self.h2n1 = Neuron(weights, bias)   #Second Hidden Layer
        self.h2n2 = Neuron(weights, bias)
        self.h2n3 = Neuron(weights, bias)
        self.h2n4 = Neuron(weights, bias)

        self.o1 = Neuron(weights, bias)     #Output Neuron 

    def feedforward(self, x):
        #outputs of the first hidden layer
        out_h1 = self.h1n1.feedforward(x)
        out_h2 = self.h1n2.feedforward(x)
        out_h3 = self.h1n3.feedforward(x)
        out_h4 = self.h1n4.feedforward(x)

        # outputs from the first hidden layer
        h1_outputs = np.array([out_h1, out_h2, out_h3, out_h4])

        # outputs of the second hidden layer
        out_h1_2 = self.h2n1.feedforward(h1_outputs)
        out_h2_2 = self.h2n2.feedforward(h1_outputs)
        out_h3_2 = self.h2n3.feedforward(h1_outputs)
        out_h4_2 = self.h2n4.feedforward(h1_outputs)

        #outputs from the second hidden layer
        h2_outputs = np.array([out_h1_2, out_h2_2, out_h3_2, out_h4_2])

        #output neuron
        out_o1 = self.o1.feedforward(h2_outputs)

        return out_o1

networksys = Neuron_Network()
x = np.array([1,0,2,0])
print(networksys.feedforward(x))