#!/usr/bin/python3

import sys
import json
import copy
import numpy as np
import pandas as pd
from random import random

INPUT_SIZE = 4
OUTPUT_SIZE = 3
LAYERS_DIMENSION = 4
NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_EPOCHS = 100
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
        self.weights.append(np.array(2*np.random.random((INPUT_SIZE, LAYERS_DIMENSION)) -1))
        self.weights.append(np.array(2*np.random.random((LAYERS_DIMENSION, LAYERS_DIMENSION)) -1))
        self.weights.append(np.array(2*np.random.random((LAYERS_DIMENSION, OUTPUT_SIZE)) -1))


    def get_weights(self):
        with open('weights.json', 'rb') as fp:
            self.weights = json.load(fp)

    def save_weights(self):
        with open("weights.json", "w") as fp:
            json.dump(self.weights, fp)

    def get_random_biases(self):
        self.biases = []
        self.biases.append(np.random.uniform(low=0, high=1, size=(LAYERS_DIMENSION)))
        self.biases.append(np.random.uniform(low=0, high=1, size=(LAYERS_DIMENSION)))
        self.biases.append(np.random.uniform(low=0, high=1, size=(OUTPUT_SIZE)))
        # self.biases.append(np.zeros(LAYERS_DIMENSION))
        # self.biases.append(np.zeros(LAYERS_DIMENSION))
        # self.biases.append(np.zeros(OUTPUT_SIZE))
        # self.biases.append(np.array(2*np.random.random((LAYERS_DIMENSION, 1)) - 1))
        # self.biases.append(np.array(2*np.random.random((LAYERS_DIMENSION, 1)) - 1))
        # self.biases.append(np.array(2*np.random.random((OUTPUT_SIZE, 1)) - 1))

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
        
    # def softmax(self):
    #     # np.random.rand()
    #     divider = 0
    #     for value in self.output:
    #         divider += np.exp(value)
        
    #     for index, value in enumerate(self.output):
    #         self.output[index] = np.exp(value)/divider
    
    def softmax(self):
        self.output = np.exp(self.output) / np.sum(np.exp(self.output), axis=-1, keepdims=True)
        
    def softmax2(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        
    def softmax_derivative_2(self, x):
        s = self.softmax2(x[:])
        return np.diag(s) - np.outer(s, s)
            
    def softmax_derivative(self, softmax_product: list):
        max_index = np.argmax(softmax_product)
        max_value = max(softmax_product)
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
        self.layer1 += self.biases[0]
        self.relu(self.layer1)
            
        self.layer2 = np.dot(self.layer1, self.weights[1])
        self.layer2 += self.biases[1]
        self.relu(self.layer2)
        
        self.output = np.dot(self.layer2, self.weights[2])
        self.output += self.biases[2]
        self.softmax()


    def back_propagation(self, input_array, real_output):
        input_array = np.array(input_array)
        relu_layer_derivative = np.vectorize(self.relu_derivative)
        
        # output => layer2
        error = np.array(real_output) - np.array(self.output)
        # delta_output = error * self.softmax_derivative(self.output)
        test_derivative = self.softmax_derivative_2(self.output)
        delta_output = np.dot(error, test_derivative)

        delta_output_reshaped = delta_output.reshape(delta_output.shape[0], -1).T
        layer2_reshaped = self.layer2.reshape(self.layer2.shape[0],-1)
        derivative_output = np.dot(layer2_reshaped, delta_output_reshaped)
        
        error = np.dot(delta_output, self.weights[2].T)
        self.biases[2] += delta_output * LEARNING_RATE
        self.weights[2] += derivative_output * LEARNING_RATE
        
        # layer2 => layer1
        delta_layer2 = error * relu_layer_derivative(self.layer2)
        delta_layer2_reshaped = delta_layer2.reshape(delta_layer2.shape[0], -1).T
        layer1_reshaped = self.layer1.reshape(self.layer1.shape[0],-1)
        derivative_layer2 = np.dot(layer1_reshaped, delta_layer2_reshaped)
        
        error = np.dot(delta_layer2, self.weights[1].T)
        self.biases[1] += delta_layer2 * LEARNING_RATE
        self.weights[1] += derivative_layer2 * LEARNING_RATE
        
        
        # layer1 => input
        delta_layer1 = error * relu_layer_derivative(self.layer1)
        delta_layer1_reshaped = delta_layer1.reshape(delta_layer1.shape[0], -1).T
        input_reshaped = input_array.reshape(input_array.shape[0],-1)
        derivative_layer1 = np.dot(input_reshaped, delta_layer1_reshaped)
        
        self.biases[0] += delta_layer1 * LEARNING_RATE
        self.weights[0] += derivative_layer1 * LEARNING_RATE
        
        
    def measure_accuracy(self):
        success = 0
        number_of_samples = 0
        testing_data = self.upload_data('iris_test.xlsx')
        for index, _ in testing_data.iterrows():
            input_array = [testing_data.loc[index, 'Petal_width'],testing_data.loc[index,'Petal_length'],testing_data.loc[index, 'Sepal_width'],testing_data.loc[index,'Sepal_length']]
            correct_output = np.zeros(OUTPUT_SIZE)
            correct_output[testing_data.loc[index, 'Species_No'] - 1] = 1
            self.run(input_array)
            self.argmax()
            number_of_samples += 1
            if self.result() == testing_data.loc[index, 'Species_name']: success += 1
        return success/number_of_samples
        
    
    def train(self):
        for epoch in range(NUMBER_OF_EPOCHS):
            self.training_data = self.upload_data('iris_train.xlsx')
            i = 0
            for index, _ in self.training_data.iterrows():
                input_array = [self.training_data.loc[index, 'Petal_width'],self.training_data.loc[index,'Petal_length'],self.training_data.loc[index, 'Sepal_width'],self.training_data.loc[index,'Sepal_length']]
                correct_output = np.zeros(OUTPUT_SIZE)
                correct_output[self.training_data.loc[index, 'Species_No'] - 1] = 1
                self.run(input_array)
                if i == 0: print('EPOCH: ',epoch,'CORRECT OUTPUT: ', correct_output, 'NN OUTPUT: ', self.output)
                self.back_propagation(input_array, correct_output)
                i += 1
    
    def upload_data(self, file):
        data = pd.read_excel(file)
        return data.sample(frac = 1)
        

if __name__ == '__main__':
    iris_flower = []
    if sys.argv[1:]:
        iris_flower = sys.argv[1:]
    nn = Neural_Network(0)
    pw = copy.deepcopy(nn.weights)
    pb = copy.deepcopy(nn.biases)
    # print(f'PRZED TRENINGIEM: FINAL WEIGHTS: {pw}, BIASES: {pb}')
    nn.train()
    print(f'PRZED TRENINGIEM: WEIGHTS: {pw}, BIASES: {pb}')
    print(f'PO TRENINGU: FINAL WEIGHTS: {nn.weights}, BIASES: {nn.biases}')
    nn.run([0.2, 1.4, 3.3, 5]) # Sestosa
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result())
    nn.run([1.3, 4.1, 2.8, 6.7]) # versicolor
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result())
    
    nn.run([1.8, 5.1, 3, 5.9]) # Verginica
    print(nn.output)
    nn.argmax()
    print('Result: ', nn.result()) # Verginica
    # nn.save_weights()
    # nn.save_biases()
    print(f'ACCURACY OF NN={nn.measure_accuracy()}')
