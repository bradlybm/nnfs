# video 4 -> Batches, Layers and Objects
import numpy as np
# converting single batch of inputs [1.0, 2.0, 3.0, 2.5] to a batch of inputs
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
 

weights = [[0.2, 0.8, -0.5, 1.0], # matrix containing vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(weights, inputs) + bias
print(output)

# the bigger the batch size the better a neuron can fit an example set but not recommended to pass more than 32 in batch size 
# (different problems will require different batch size) 6:00
