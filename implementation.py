import data_prep as dp
import nn_source as nn
import pickle as pkl

(x_train, y_train), (x_test, y_test) = dp.MnistDataloader().return_clean_data()
                                    # this list comprehension is turning a digit into its one-hot representation
training_data: [([float], [float])] = [(x_train[i]/255, [1 if j == y_train[i] else 0 for j in range(10)]) for i in range(1000)]
test_data: [([float], [float])] = [(x_test[i]/255, [1 if j == y_test[i] else 0 for j in range(10)]) for i in range(len(x_test))]

# Inititalizing the network
network: nn.Neural_Network = nn.Neural_Network(784, 10, [(512, 128), (128, 10)])

def train_network(epochs: int, learning_rate: float):
    for epoch in range(epochs):
        print(f"Epoch: {epoch}", end="\r")
        for data_point in training_data:
            network.backpropagate(data_point, learning_rate)
        with open("serialization\\network.pkl", "wb") as file:
            pkl.dump(network, file)
    print("Average error: ",network.calculate_average_output_error(training_data[random.randint(0, len(training_data)-1)]))
    print(len(x_train))
#print(list(training_data[0][0]))
train_network(1, 0.01)

with (open("serialization\\network.pkl", "rb")) as file:
    network = pkl.load(file)
    print(network)