import torchvision
from torchvision.datasets import USPS

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST('/data/', download=True)