#!/usr/bin/python3

import sys
import json
import copy
import numpy as np
import pandas as pd
from random import random

INPUT_SIZE = 4
OUTPUT_SIZE = 3
LAYERS_DIMENSION = 6
NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_EPOCHS = 10000
LEARNING_RATE = 0.01

class Neural_Network:

    def __init__(self, weights) -> None:
        self.layer1 = np.zeros(LAYERS_DIMENSION)
        self.layer2 = np.zeros(LAYERS_DIMENSION)
        self.output = np.zeros(OUTPUT_SIZE)
        self.layers_size = LAYERS_DIMENSION
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.number_of_layers = NUMBER_OF_HIDDEN_LAYERS
        self.iris_names = ['Setosa', 'Versicolor', 'Verginica']
        self.get_random_weights()
        self.get_random_biases()

    def get_random_weights(self):
        self.weights = []
        # w = np.random.rand(INPUT_SIZE, LAYERS_DIMENSION)
        self.weights.append(np.array(2*np.random.random((INPUT_SIZE, LAYERS_DIMENSION)) -1))
        self.weights.append(np.array(2*np.random.random((LAYERS_DIMENSION, LAYERS_DIMENSION)) -1))
        self.weights.append(np.array(2*np.random.random((LAYERS_DIMENSION, OUTPUT_SIZE)) -1))

        # for _ in range(self.layers_size):
        #     weights_list = 2*np.random.random((self.input_size, 1)) - 1
        #     weights_list = weights_list.tolist()
        #     for index, value in enumerate(weights_list): weights_list[index] = value[0] 
        #     self.weights[0].append(weights_list)

        # for _ in range(self.layers_size):
        #     weights_list = 2*np.random.random((self.layers_size, 1)) - 1
        #     weights_list = weights_list.tolist()
        #     for index, value in enumerate(weights_list): weights_list[index] = value[0] 
        #     self.weights[1].append(weights_list)
            
        # for _ in range(self.output_size):
        #     weights_list = 2*np.random.random((self.layers_size, 1)) - 1
        #     weights_list = weights_list.tolist()
        #     for index, value in enumerate(weights_list): weights_list[index] = value[0] 
        #     self.weights[2].append(weights_list)

    def get_weights(self):
        with open('weights.json', 'rb') as fp:
            self.weights = json.load(fp)

    def save_weights(self):
        with open("weights.json", "w") as fp:
            json.dump(self.weights, fp)

    def get_random_biases(self):
        self.biases = []
        self.biases.append(np.zeros(LAYERS_DIMENSION)+1)
        self.biases.append(np.zeros(LAYERS_DIMENSION)+1)
        self.biases.append(np.zeros(OUTPUT_SIZE)+1)
        # self.biases.append(np.array(2*np.random.random((LAYERS_DIMENSION, 1)) - 1))
        # self.biases.append(np.array(2*np.random.random((LAYERS_DIMENSION, 1)) - 1))
        # self.biases.append(np.array(2*np.random.random((OUTPUT_SIZE, 1)) - 1))
        
        
        # self.biases = [[], [], []]

        # for layer in range(self.number_of_layers):
        #     biases_list = 2*np.random.random((self.layers_size, 1)) - 1
        #     biases_list = biases_list.tolist()
        #     for index, value in enumerate(biases_list): biases_list[index] = value[0] 
        #     self.biases[layer]=biases_list
        
        # biases_list = 2*np.random.random((self.output_size, 1)) - 1
        # biases_list = biases_list.tolist()  
        # for index, value in enumerate(biases_list): biases_list[index] = value[0] 
        # self.biases[2]=biases_list

    def get_biases(self):
        with open('biases.json', 'rb') as fp:
            self.biases = json.load(fp)

    def save_biases(self):
        with open("biases.json", "w") as fp:
            json.dump(self.biases, fp)
            
    def relu(self, layer: list):
        for index, value in enumerate(layer):
            layer[index] = max(0, value)
            
    def relu_derivative(self, x):
        return 1 * (x > 0)
    
    def argmax(self):
        one = np.argmax(self.output)
        self.output = np.zeros(len(self.output)).tolist()
        self.output[one] = 1
        
    def softmax(self):
        # np.random.rand()
        divider = 0
        for value in self.output:
            divider += np.exp(value)
        
        for index, value in enumerate(self.output):
            self.output[index] = np.exp(value)/divider
            
    def softmax_derivative(self, softmax_product: list):
        # TODO: use np.vectorize()
        max_index = np.argmax(softmax_product)
        max_value = max(softmax_product)
        # print('MAXXXX',max_value, max_index)
        derivative = []
        for index, value in enumerate(softmax_product):
            if index == max_index: derivative.append(max_value*(1-value))
            else: derivative.append(max_value*value)
        return np.array(derivative)
    
    def result(self):
        return self.iris_names[self.output.index(max(self.output))]
        

    def run(self, inputs: list):
        inputs = np.array(inputs)
        
        self.layer1 = np.array(np.dot(inputs, self.weights[0]))
        print(self.biases[0], self.layer1)
        self.layer1 += self.biases[0]
        self.relu(self.layer1)
            
        self.layer2 = np.dot(self.layer1, self.weights[1])
        self.layer2 += self.biases[1]
        self.relu(self.layer2)
        
        self.output = np.dot(self.layer2, self.weights[2])
        self.output += self.biases[2]
        self.softmax()
        
        # print('INPUTS: ', inputs)
        # print('HIDDEN LAYER 1: ', self.layer1, type(self.layer1))
        # print('HIDDEN LAYER 2: ', self.layer2, type(self.layer1))
        # print('OUTPUT LAYER:', self.output, type(self.output))
        # print('WEIGHTS: ', self.weights, 'BIASES: ', self.biases)

    def back_propagation(self, input_array, real_output):
        input_array = np.array(input_array)
        relu_layer_derivative = np.vectorize(self.relu_derivative)
        
        # output => layer2
        error = np.array(real_output) - np.array(self.output)
        # print(f'error on output {error}')
        delta_output = error * self.softmax_derivative(self.output)
        """
        delta_re = delta.reshape(delta.shape[0], -1).T
        current_activations = current_activations.reshape(current_activations.shape[0],-1)
        self.derivatives[i] = np.dot(current_activations, delta_re)
        """
        delta_output_reshaped = delta_output.reshape(delta_output.shape[0], -1).T
        layer2_reshaped = self.layer2.reshape(self.layer2.shape[0],-1)
        # print('delta', delta_output, 'layer2', self.layer2, self.layer2.shape)
        # print('delta re', delta_output_reshaped, 'layer2 re', layer2_reshaped)
        derivative_output = np.dot(layer2_reshaped, delta_output_reshaped)
        # print('OUTPUT DERIVATIVE: ', derivative_output, 'WEIGHTS [2]', self.weights[2])
        
        error = np.dot(delta_output, self.weights[2].T)
        self.biases[2] += delta_output * LEARNING_RATE
        self.weights[2] += derivative_output * LEARNING_RATE
        
        # print(f'error on layer2 {error}')
        # layer2 => layer1
        print('LAYER 2',self.layer2)
        print('LAYER 2 DERIVATIVE', relu_layer_derivative(self.layer2))
        delta_layer2 = error * relu_layer_derivative(self.layer2)
        delta_layer2_reshaped = delta_layer2.reshape(delta_layer2.shape[0], -1).T
        layer1_reshaped = self.layer1.reshape(self.layer1.shape[0],-1)
        derivative_layer2 = np.dot(layer1_reshaped, delta_layer2_reshaped)
        # print(f'derivatives or layer2 {derivative_layer2}')
        
        error = np.dot(delta_layer2, self.weights[1].T)
        
        self.biases[1] += delta_layer2 * LEARNING_RATE
        self.weights[1] += derivative_layer2 * LEARNING_RATE
        
        
        # print(f'error on layer1 {error}')
        # layer1 => input
        delta_layer1 = error * relu_layer_derivative(self.layer1)
        delta_layer1_reshaped = delta_layer1.reshape(delta_layer1.shape[0], -1).T
        input_reshaped = input_array.reshape(input_array.shape[0],-1)
        derivative_layer1 = np.dot(input_reshaped, delta_layer1_reshaped)
        # print(f'derivatives or layer1 {derivative_layer1}')
        self.biases[0] += delta_layer1 * LEARNING_RATE
        self.weights[0] += derivative_layer1 * LEARNING_RATE
        
        
        
        
    
    def train(self):
        # self.upload_training_data()
        # print(self.training_data)
        for epoch in range(NUMBER_OF_EPOCHS):
            self.upload_training_data()
            for index, _ in self.training_data.iterrows():
                input_array = [self.training_data.loc[index, 'Petal_width'],self.training_data.loc[index,'Petal_length'],self.training_data.loc[index, 'Sepal_width'],self.training_data.loc[index,'Sepal_length']]
                correct_output = np.zeros(OUTPUT_SIZE)
                correct_output[self.training_data.loc[index, 'Species_No'] - 1] = 1
                # print('training data',self.training_data.loc[index], 'output:', correct_output)
                self.run(input_array)
                print('EPOCH: ',epoch,'CORRECT OUTPUT: ', correct_output, 'NN OUTPUT: ', self.output)
                if index == 1: print(self.weights)
                self.back_propagation(input_array, correct_output)
                if index > 30: break
    
    def upload_training_data(self):
        self.training_data = pd.read_excel('iris.xlsx')
        self.training_data = self.training_data.sample(frac = 1)
        # print(self.training_data)
        # self.training_data = self.training_data.sample(frac = 1)
        

if __name__ == '__main__':
    iris_flower = []
    if sys.argv[1:]:
        iris_flower = sys.argv[1:]
    nn = Neural_Network(0)
    # print('nowe wagi', nn.weights)
    # nn.get_weights()
    # nn.get_biases()
    # nn.run([1.2, 3.4, 0.6, 2.5])
    # print(nn.output)
    # nn.softmax()
    # print('po softmaxie', nn.output)
    # print('po derywatywie', nn.softmax_derivative(nn.output), type(nn.softmax_derivative(nn.output)))
    pw = copy.deepcopy(nn.weights)
    pb = copy.deepcopy(nn.weights)
    # print(f'PRZED TRENINGIEM: FINAL WEIGHTS: {pw}, BIASES: {pb}')
    nn.train()
    print(f'PRZED TRENINGIEM: FINAL WEIGHTS: {pw}, BIASES: {pb}')
    print(f'PO TRENINGU: FINAL WEIGHTS: {nn.weights}, BIASES: {nn.biases}')
    nn.run([0.3, 1.3, 3.5, 5]) # Sestosa
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result())
    nn.run([1.4,4.7,2.9,6.1]) # versicolor
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result())
    
    nn.run([2.3,5.3,3.2,6.4]) # versicolor
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result()) # Verginica
    # nn.save_weights()
    # nn.save_biases()
