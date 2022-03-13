import torchvision
import os
version = "1.0"

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(os.getcwd() + "/files/MNIST/", download=True)