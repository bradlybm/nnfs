#video 1 
#coding 1 neuron sommewhere in a neural network
#3 neurons are going to be feeding into it

inputs = [1.2, 5.1, 2.1] #3 unique outputs from the 3 neurons of the previous layer
weights = [3.1, 2.1, 8.7] #every unnique input is going to have a weight associated with it
bias = 3 #every unique neuron is going to have a unique bias

#first step for a neuron is to add up all the inputs * weight + bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

