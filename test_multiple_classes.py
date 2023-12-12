from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import nn_source as nn
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def train_network(network, training_data: [([float], [float])], learning_rate: float, epochs: int):
    for i in range(epochs):
        print("Epoch: ", i, end="\r")
        for data_point in training_data:
            network.backpropagate(data_point, learning_rate)
        


def test_network(network, test_data: [([float], [float])]):
    for data_point in test_data:
        print("Expected: ", data_point[1])
        print("Actual: ", list(network.calculate_outputs(np.array(data_point[0]))))
        print("\n")

def main():
    # create training data

    #X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_classes=2, random_state=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Create a multi-label classification dataset
    X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=3, n_labels=2, random_state=1)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #print("Sample:", list(X_train[0]))
    #print("Labels:", list(y_train[0]))
    #print("TEST:", X_test[0], y_test[0])
    training_data: [([float], [float])] = [(list(X_train[i]), list(y_train[i])) for i in range(len(X_train))]
    test_data: [([float], [float])] = [(list(X_test[i]), list(y_test[i])) for i in range(len(X_test))]


    # create neural network
    network = nn.Neural_Network(20, 3, [(100, 50), (50, 3)])
    #network = nn.Neural_Network(10, 3, [(50, 20), (20, 3)])


    # train neural network
    train_network(network, training_data, 0.01, 100)
    #train_network_with_mini_batches(network, training_data, 0.1, 100, 10)

    # test neural network
    test_network(network, test_data)



main()