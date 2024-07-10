# video 4 -> Batches, Layers and Objects
import numpy as np
# converting single batch of inputs [1.0, 2.0, 3.0, 2.5] to a batch of inputs
"""
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
 

weights = [[0.2, 0.8, -0.5, 1.0], # matrix containing vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(weights, inputs) + bias
print(output)
"""
# the weight in the code above needs to be transposed in order for the .dot product to be made (10:20)

# the bigger the batch size the better a neuron can fit an example set but not recommended to pass more than 32 in batch size 
# (different problems will require different batch size) 6:00

# transpose swaps rows and columns 

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
 

weights = [[0.2, 0.8, -0.5, 1.0], # matrix containing vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# adding a second layer
weights2 = [[0.1, -0.14, 0.5], # matrix containing vectors
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]



layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_output = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_output)


"""need to convert weights to np array so i can use numpy to transpose
   also needed to switch weights and inputs around because """

# 13:33 for reference below
"""
Transposing the weights will make it go from looking like this:
[0.2,    0.8, -0.5,   1.0]
[0.5,   -0.91, 0.26, -0.5]
[-0.26, -0.27, 0.17,  0.87]
"""
"""
to making the weights look like this:
[0.2,  0.5,  -0.26]
[0.8, -0.91, -0.27]
[-0.5, 0.26,  0.17]
[1.0, -0.5,   0.87]
"""

