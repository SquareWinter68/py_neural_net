import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import nn_source as nn
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

def train_network(network, training_data: [([float], [float])], learning_rate: float, epochs: int):
    for i in range(epochs):
        print("Epoch: ", i)
        for data_point in training_data:
            network.backpropagate(data_point, learning_rate)
        print("Average error: ",network.calculate_average_output_error(training_data[random.randint(0, len(training_data)-1)]))
        print("\n")

def train_network_with_mini_batches(network, training_data: [([float], [float])], learning_rate: float, epochs: int, batch_size: int):
    for i in range(epochs):
        print("Epoch: ", i)
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
        for mini_batch in mini_batches:
            network.mini_batch_gradient_descent(mini_batch, learning_rate*(1/(i+1)))
        print("\n")
        print("Average error: ",network.calculate_average_output_error(training_data[random.randint(0, len(training_data)-1)]))

# test training data
def test_network(network, test_data: [([float], [float])]):
    for data_point in test_data:
        print("Expected: ", data_point[1])
        print("Actual: ", list(network.calculate_outputs(np.array(data_point[0]))))
        #print("Actual softmax: ", list(softmax(network.calculate_outputs(np.array(data_point[0])))))
        print("\n")


def main():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,  # Number of samples
        n_features=20,   # Number of features
        n_classes=2,     # Number of classes (binary classification)
        random_state=42
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_data: [([float], [float])] = [(list(X_train[i]), [y_train[i]]) for i in range(len(X_train))]
    test_data: [([float], [float])] = [(list(X_test[i]), [y_test[i]]) for i in range(len(X_test))]



    network = nn.Neural_Network(20, 1, [(20, 10), (10, 1)])

    train_network(network, training_data, 0.1, 50)

    test_network(network, test_data)



def three_classes():
    np.random.seed(42)

    # Generate a synthetic classification dataset with three classes
    X, y = make_classification(
        n_samples=5000,   # Number of samples
        n_features=20,    # Number of features
        n_classes=3,      # Number of classes (change to desired number)
        n_clusters_per_class=1,  # Number of clusters per class
        random_state=42
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_data: [([float], [float])] = [(list(X_train[i]), [1 if j == y_train[i] else 0 for j in range(3)]) for i in range(len(X_train))]
    test_data: [([float], [float])] = [(list(X_test[i]), [1 if j == y_train[i] else 0 for j in range(3)]) for i in range(len(X_test))]



    network = nn.Neural_Network(20, 3, [(3, 3)])

    train_network(network, training_data, 0.1, 50)
    #train_network_with_mini_batches(network, training_data, 0.1, 100, 100)

    test_network(network, test_data)


#main()
# Print the shapes of the datasets
three_classes()
""" print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape) """