import random

import numpy as np
import math
from typing import List, Tuple

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from tensorflow.keras.datasets import mnist

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.inputs: np.ndarray[np.float64] = None
        self.biases: np.ndarray[np.float64] = None
        self.weights: np.ndarray[np.float64] = None
        self.weighted_sum: np.ndarray[np.float64] = None
        self.activations: np.ndarray[np.float64] = None
        self.bias_gradient_mean: np.ndarray[np.float64] = np.array([0.0]*output_size)
        self.weight_gradient_mean: np.ndarray[np.float64] =  np.zeros((output_size, input_size))
        self.initialize_weights()
        self.initialize_biases()

    def initialize_biases(self):
        #Biases are column vectors by dfault , which is why the np_array is transposed at the end
        self.biases = np.random.rand(self.output_size).T

    def initialize_weights(self):
        #Weights are matrices of shape output x input
        self.weights = np.random.rand(self.output_size, self.input_size)

    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def calculate_outputs(self, inputs: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        activation_func = np.vectorize(self.sigmoid)
        self.weighted_sum = (self.weights @ inputs.T) + self.biases 
        return activation_func(self.weighted_sum) 
        # VERIFIED WORKS

    def update_weights_and_biases(self, weights: np.ndarray[np.float64], biases: np.ndarray[np.float64], learning_rate: float):
        self.weights = self.weights - (weights * learning_rate)
        self.biases = self.biases - (biases * learning_rate)

class Neural_Network:
    def __init__(self, input_layer_size:int, output_layer_size:int, hidden_layers:[(int, int)]):
        if hidden_layers:
            self.layers: [Layer] = [Layer(input_layer_size, hidden_layers[0][0]) if i == 0 else Layer(hidden_layers[i-1][0], hidden_layers[i-1][1]) for i in range(len(hidden_layers) + 1)]
        else:
            self.layers: [Layer] = [Layer(input_layer_size, output_layer_size)]

    def sigmoid_derivative(self, xs : np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        sigmoid : np.ndarray[np.float64] = np.vectorize(lambda x: 1/(1 + (math.e**(-x))))
        # sigmoid derivative da/dx = sigmoid*(1-sigmoid)
        return sigmoid(xs) * (1 - sigmoid(xs))
        # VERIFIED WORKS

    def calculate_outputs(self, inputs:np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        for layer in self.layers:
            layer.inputs = inputs
            inputs = layer.calculate_outputs(inputs)
        return inputs
        # VERIFIED WORKS

    def calculate_average_output_error(self, data_point: ([float], [float])):
        print(self.calculate_outputs(np.array(data_point[0])))
        # subtracting expected values from calculated
        result: np.ndarray[np.float64] = self.calculate_outputs(np.array(data_point[0]))
        result = result - np.array(data_point[1])
        return (result @ result)/len(data_point[1])
        # VERIFIED WORKS

    def calculate_output_error_gradient(self, data_point: ([float], [float])):
        # Bias gradient / error given by: dc/da * da/dx * dx/db1     x = i1*w1 + i2*w2 ... + b1 => dx/db1 = 1
        # dc/da * da/dx *1,  dc/da = 2(activation - out),   da/dx = sigmoid*(1-sigmoid)

        outputs: np.ndarray[np.float64] = self.calculate_outputs(np.array(data_point[0]))

        # Calculating 2(activation - out)
        dc_da : np.ndarray[np.float64] = 2 * (outputs - np.array(data_point[1]))

        # Calculating sigmoid*(1-sigmoid)
        da_dx : np.ndarray[np.float64] = outputs * ((outputs * -1) + 1)

        #intermediary step_for validation
        result : np.ndarray[np.float64] = dc_da * da_dx
        return result.T
        # VERIFIED WORKS

    def calculate_error_gradient(self, next_layer_error : np.ndarray[np.float64], next_layer_weights : np.ndarray[np.float64], weighted_sums : np.ndarray[np.float64]):
        #δl=((wl+1)Tδl+1)⊙σ′(zl)

        delta: np.ndarray[np.float64] = next_layer_weights.T @ next_layer_error * (self.sigmoid_derivative(weighted_sums))
        return delta
        # VERIFIED WORKS

    def backpropagate(self, data_point: ([float], [float]), learning_rate: float):
        # Calculate output error gradient
        error : np.ndarray[np.float64] = self.calculate_output_error_gradient(data_point)
        # Formula for calculating weight gradient:
        # Errors (column vector) x Inputs (row vector)
        self.layers[-1].update_weights_and_biases(np.outer(error, self.layers[-1].inputs), error, learning_rate)

        
        for i in range(len(self.layers) -2, -1, -1):
            # Backpropagate error through layers
            error = self.calculate_error_gradient(error, self.layers[i+1].weights, self.layers[i].weighted_sum)

            self.layers[i].update_weights_and_biases(np.outer(error, self.layers[i].inputs), error, learning_rate)

    def backpropagate_for_mini_batch(self, data_point: ([float], [float]), learning_rate: float):
        error : np.ndarray[np.float64] = self.calculate_output_error_gradient(data_point)
        # Formula for calculating weight gradient:
        # Errors (column vector) x Inputs (row vector)
        self.layers[-1].bias_gradient_mean += error
        self.layers[-1].weight_gradient_mean += np.outer(error, self.layers[-1].inputs)
        
        for i in range(len(self.layers) -2, -1, -1):
            error = self.calculate_error_gradient(error, self.layers[i+1].weights, self.layers[i].weighted_sum)

            self.layers[i].bias_gradient_mean += error
            self.layers[i].weight_gradient_mean += np.outer(error, self.layers[i].inputs)
    
    def mini_batch_gradient_descent(self, mini_batch: [([float], [float])], learning_rate: float):
        for data_point in mini_batch:
            self.backpropagate_for_mini_batch(data_point, learning_rate)
        for layer in self.layers:
            layer.update_weights_and_biases(layer.weight_gradient_mean/len(mini_batch), layer.bias_gradient_mean/len(mini_batch), learning_rate)
            layer.weight_gradient_mean = np.zeros(layer.weight_gradient_mean.shape)
            layer.bias_gradient_mean = np.zeros(layer.bias_gradient_mean.shape)



# train my neural network
def train_network(network: Neural_Network, training_data: [([float], [float])], learning_rate: float, epochs: int):
    for i in range(epochs):
        print("Epoch: ", i)
        for data_point in training_data:
            network.backpropagate(data_point, learning_rate)
        print("\n")

def train_network_with_mini_batches(network: Neural_Network, training_data: [([float], [float])], learning_rate: float, epochs: int, batch_size: int):
    for i in range(epochs):
        print("Epoch: ", i)
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
        for mini_batch in mini_batches:
            network.mini_batch_gradient_descent(mini_batch, learning_rate)
        print("\n")

# test training data
def test_network(network: Neural_Network, test_data: [([float], [float])]):
    for data_point in test_data:
        print("Expected: ", data_point[1])
        print("Actual: ", network.calculate_outputs(np.array(data_point[0])))
        print("\n")

def main():
    # create training data

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_classes=2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("TEST:", X_test[0], y_test[0])
    training_data: [([float], [float])] = [(X_train[i], [y_train[i]]) for i in range(len(X_train))]
    test_data: [([float], [float])] = [(X_test[i], [y_test[i]]) for i in range(len(X_test))]

    # create neural network
    network: Neural_Network = Neural_Network(10, 1, [(10, 10), (10, 1)])

    # train neural network
    train_network(network, training_data, 0.1, 100)
    #train_network_with_mini_batches(network, training_data, 0.1, 100, 10)

    # test neural network
    test_network(network, test_data)

main()

# q: How to ubnstage files in git?
