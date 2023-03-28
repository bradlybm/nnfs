# video 3 -> the dot product
# simlyfing the code

 # inputs = [1, 2, 3, 2.5] 

# weights = [[0.2, 0.8, -0.5, 1.0], # weight1
#            [0.5, -0.91, 0.26, -0.5], # weight2
#            [-0.26, -0.27, 0.17, 0.87]] # weight3

# biases = [2, 3, 0.5] # bias 1, 2, 3

# layer_outputs = [] # out of current layer
# for neuron_weights, neuron_bias in zip(weights, biases): # zip() is multiplying weights and biases (I think)
#     neuron_output = 0 # output of given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

 # print(layer_outputs)

# Python's zip() function creates an iterator that will aggregate (put together) elements from two or more iterables.
    # An iterable is any Python object capable of returning its members one at a time
# *Note: weighted sum is all the inputs multiplied by the wights then added all up including the biases

# dot_product ex:
# a = [1, 2, 3]
# b = [2, 3, 4]
# dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
# >> 20
# using dot product below

# import numpy as np

# inputs = [1.0, 2.0, 3.0, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2

# output = np.dot(weights, inputs) + bias
# print(output)

# np.dot is multiplying the elements in the vectors using the index as 
# ex: = 0.2 * 1.0 + 0.8 * 2.0 + -0.5 * 3.0 + 1.0 * 2.5
#     = 2.8
#     = 2.8 + bias = 4.8


# dot product example of a layer of neurons below
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5] # vector

weights = [[0.2, 0.8, -0.5, 1.0], # matrix containing vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(weights, inputs) + bias
print(output)

