import data_prep as dp
import nn_source as nn
import pickle as pkl
import numpy as np
import time
(x_train, y_train), (x_test, y_test) = dp.MnistDataloader().return_clean_data()
                                    # this list comprehension is turning a digit into its one-hot representation
training_data: [([float], [float])] = [(x_train[i], [1 if j == y_train[i] else 0 for j in range(10)]) for i in range(1000)]
test_data: [([float], [float])] = [(x_test[i], [1 if j == y_test[i] else 0 for j in range(10)]) for i in range(len(x_test))]

def test_network(network: nn.Neural_Network, test_data: [([float], [float])]):
    for data_point in test_data:
        print("Expected: ", data_point[1])
        print("Actual: ", list(network.calculate_outputs(np.array(data_point[0]))))
        print("\n")
        time.sleep(1)


with (open("serialization\\network134.pkl", "rb")) as file:
    network = pkl.load(file)
    print(network)
    test_network(network, test_data)