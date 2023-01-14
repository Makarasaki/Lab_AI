#!/usr/bin/python3

import sys
import json
import random
import numpy as np

INPUT_SIZE = 4
OUTPUT_SIZE = 3
LAYERS_DIMENSION = 6
NUMBER_OF_HIDDEN_LAYERS = 2


class Neural_Network:

    def __init__(self, weights) -> None:
        self.layer1 = np.zeros(LAYERS_DIMENSION)
        self.layer2 = np.zeros(LAYERS_DIMENSION)
        self.output = np.zeros(OUTPUT_SIZE)
        self.weights = []
        self.biases = []
        self.layers_size = LAYERS_DIMENSION
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.number_of_layers = NUMBER_OF_HIDDEN_LAYERS

    def get_random_weights(self):
        self.weights = [[], [], []]

        for _ in range(self.layers_size):
            weights_list = 2*np.random.random((self.input_size, 1)) - 1
            weights_list = weights_list.tolist()
            for index, value in enumerate(weights_list): weights_list[index] = value[0] 
            self.weights[0].append(weights_list)

        for _ in range(self.layers_size):
            weights_list = 2*np.random.random((self.layers_size, 1)) - 1
            weights_list = weights_list.tolist()
            for index, value in enumerate(weights_list): weights_list[index] = value[0] 
            self.weights[1].append(weights_list)
            
        for _ in range(self.output_size):
            weights_list = 2*np.random.random((self.layers_size, 1)) - 1
            weights_list = weights_list.tolist()
            for index, value in enumerate(weights_list): weights_list[index] = value[0] 
            self.weights[2].append(weights_list)

    def get_weights(self):
        with open('weights.json', 'rb') as fp:
            self.weights = json.load(fp)

    def save_weights(self):
        with open("weights.json", "w") as fp:
            json.dump(self.weights, fp)

    def get_random_biases(self):
        self.biases = [[], [], []]

        for layer in range(self.number_of_layers):
            biases_list = 2*np.random.random((self.layers_size, 1)) - 1
            biases_list = biases_list.tolist()
            for index, value in enumerate(biases_list): biases_list[index] = value[0] 
            self.biases[layer]=biases_list
        
        biases_list = 2*np.random.random((self.output_size, 1)) - 1
        biases_list = biases_list.tolist()  
        for index, value in enumerate(biases_list): biases_list[index] = value[0] 
        self.biases[2]=biases_list

    def get_biases(self):
        with open('biases.json', 'rb') as fp:
            self.biases = json.load(fp)

    def save_biases(self):
        with open("biases.json", "w") as fp:
            json.dump(self.biases, fp)
            
    def relu(self, layer: list):
        for index, value in enumerate(layer):
            layer[index] = max(0, value)
        

    def run(self, inputs: list):
        for neuron in range(self.layers_size):
            self.layer1[neuron] = np.dot(inputs, self.weights[0][neuron])
        
        for neuron in range(self.layers_size):
            self.layer1[neuron] += self.biases[0][neuron]
            
        nn.relu(self.layer1)
            
        for neuron in range(self.layers_size):
            self.layer2[neuron] = np.dot(self.layer1, self.weights[1][neuron])
        
        for neuron in range(self.layers_size):
            self.layer2[neuron] += self.biases[1][neuron]
            
        nn.relu(self.layer2)
        
        for neuron in range(self.output_size):
            self.output[neuron] = np.dot(self.layer2, self.weights[2][neuron])
        
        for neuron in range(self.output_size):
            self.output[neuron] += self.biases[1][neuron]
            
        print('HIDDEN LAYER 1: ', self.layer1)
        print('HIDDEN LAYER 2: ', self.layer2)
        print('OUTPUT LAYER:', self.output)

    def train():
        pass


# def relu(x):
#     return np.maximum(0, x)


# def random_weights():
#     weights = []
#     for _ in range(2*INPUT_SIZE*LAYERS_DIMENSION):
#         weights.append(random.uniform(0, 1))
#     print(weights)
#     return weights


# def neural_network(input: list, weights: list):
#     layer1_values = []
#     layer2_values = []


# def save_weights(a_list):
#     with open("weights.json", "w") as fp:
#         json.dump(a_list, fp)


# def get_weights():
#     with open('weights.json', 'rb') as fp:
#         nn_weights = json.load(fp)
#         return nn_weights


if __name__ == '__main__':
    iris_flower = []
    if sys.argv[1:]:
        iris_flower = sys.argv[1:]
    nn = Neural_Network(0)
    # nn.get_random_weights()
    # nn.get_random_biases()
    nn.get_weights()
    nn.get_biases()
    nn.run([1.2, 3.4, 0.6, 2.5])
    nn.save_weights()
    nn.save_biases()
