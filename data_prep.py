import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
import random
import nn_source as nn
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    input_path = r'C:\Users\Vukasin\Desktop\nn_python\too_much_stuff'
    def __init__(self, training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'), test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'), test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

    def return_clean_data(self) -> ([float], [float]):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        x_train = [np.array(x).flatten() for x in x_train]
        x_test = [np.array(x).flatten() for x in x_test]
        return (x_train, y_train), (x_test, y_test)        

class Visualize:
    def __init__(self):
        self.input_path = r'C:\Users\Vukasin\Desktop\nn_python\too_much_stuff'
        self.training_images_filepath = join(self.input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        self.training_labels_filepath = join(self.input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        self.test_images_filepath = join(self.input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        self.test_labels_filepath = join(self.input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        self.images_to_show = []
        self.labbels_to_show = []
        #load mnist data
        print("Started loading....")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = MnistDataloader(self.training_images_filepath, self.training_labels_filepath, self.test_images_filepath, self.test_labels_filepath).load_data()
        print("Finnished_loading")
    def show_images(self):
        cols = 5
        rows = int(len(self.images_to_show)/cols) + 1    
        plt.figure(figsize=(30,20))
        for i in range(len(self.images_to_show)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(self.images_to_show[i], cmap=plt.cm.gray)
            plt.title(self.labbels_to_show[i])
        plt.show()
    
    def load_random_images(self):
        # The number in thr range is the number of images to show
        for i in range(10):
            index: int = random.randint(0, len(self.x_train))
            self.images_to_show.append(self.x_train[index])
            self.labbels_to_show.append(f"training image {index}, Value :{self.y_train[index]}")

#input_path = r'C:\Users\Vukasin\Desktop\nn_python\too_much_stuff'
#training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
#training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
#test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
#test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')



""" def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1    
    plt.figure(figsize=(30,20))
    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(title_texts[i])
    plt.show()

print("Started loading....")
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print("Finnished_loading")

images_to_show = []
labbels_to_show = [] """

""" def load_random_images():
    # The number in thr range is the number of images to show
    for i in range(10):
        rnd: int = random.randint(0, len(x_train))
        index = random.randint(0, len(x_train))
        images_to_show.append(x_train[rnd])
        labbels_to_show.append(f"training image {rnd}, Value :{y_train[i]}") """

#load_random_images()
#show_images(images_to_show, labbels_to_show)

""" def train_network(network: nn.Neural_Network, training_data: [([float], [float])], learning_rate: float, epochs: int):
    for i in range(epochs):
        print(f"Epoch: {i}", end="\r")
        for data_point in training_data:
            network.backpropagate(data_point, learning_rate)
        
#flatten data

x_train = [np.array(x).flatten() for x in x_train]
x_test = [np.array(x).flatten() for x in x_test]

print(len(x_train[0]))
print(y_train[0])
network: nn.Neural_Network = nn.Neural_Network(784, 10, [(784, 300), (300, 10)])
training_data: [([float], [float])] = [(x_train[i], [y_train[i]]) for i in range(1000)]
test_data: [([float], [float])] = [(x_test[i], [y_test[i]]) for i in range(len(x_test))]
#train_network(network, training_data, 0.1, 100)


# This assertion is true meaning the return_clean_data() method works reliably
np.testing.assert_array_equal(x_train, mnist_dataloader.return_clean_data()[0][0]) """
