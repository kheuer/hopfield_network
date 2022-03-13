import torchvision
import os

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(os.getcwd() + "/files/MNIST/", download=True)