# video 2 -> modeling an output layer (3 neurons with 4 inputs)
# coding 1 neuron sommewhere in a neural network
# 4 neurons are going to be feeding into it

# becasue I am modeling the output layer that will take 4 inputs form 4 different neurons
# we need 3 unique weight sets and 3 unique weight sets becasue we are modeling the 3 neuron output layer

inputs = [1, 2, 3, 2.5] 

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2 
bias2 = 3
bias3 = 0.5

# beacuse I am modeling the output layer (3 different neurons) the output will look like the input (except only with 3 values)
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)

# In order to change the output value I can the weights and biases (which is the struggle of NN and ML...intersting)
