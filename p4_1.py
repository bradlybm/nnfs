"""continuation of p4.py
going to be using the same inputs weights and biases (layers)
but I will make them into objects in this file
(aka simplifying the code)
"""

import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense: # 21:06
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5) # the output of this (5)
layer2 = Layer_Dense(5, 2) # has to be the input of this however the output of layer 2 can be any number i want.

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


