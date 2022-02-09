import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import logging
logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HopfieldNetwork:
    def __init__(self, n_neurons):
        if n_neurons < 4:
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be at least 4")
        sqrt = np.sqrt(n_neurons)
        if not sqrt.is_integer():
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be divisible by itself to an int")
        self.n_neurons = n_neurons
        self.size = (int(sqrt), int(sqrt))
        self.patterns = []
        self.state = None
        self.set_random_state()
        logger.info(f"Initialized Hopfield network of size {self.size} with {self.n_neurons} Neurons")

    def train(self, pattern):
        # todo
        self.patterns.append(pattern)


    def visualize(self, state):
        fig = Figure(figsize=(3, 3), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(state, cmap="Blues",  interpolation="nearest")
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        return fig

    def get_random_state(self):
        return np.random.rand(self.size[0], self.size[1]).round()

    def set_state(self, state):
        self.state = state

    def set_random_state(self):
        self.state = self.get_random_state()

    def visualize_pattern(self, i):
        return self.visualize(self.patterns[i])

if __name__ == '__main__':
    nn = HopfieldNetwork(100)
    
