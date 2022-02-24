import torch
from torch import Tensor, nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageDraw, ImageFont

import logging
logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HopfieldNetwork:
    def __init__(self, n_neurons):
        """
        Initialize a Hopfield Network with a quadratic shape of (n_neurons**0.5, n_neurons**0.5)

        :param n_neurons: The number of neurons the network should have.
        """
        if n_neurons < 4:
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be at least 4")
        sqrt = np.sqrt(n_neurons)
        if not sqrt.is_integer():
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be divisible by itself to an int")
        if n_neurons < 100:
            logger.warning("We recommend to choose n_neurons to be >= 100 to ensure proper generation of characters.")
        self.n_neurons = n_neurons
        self.size = (int(sqrt), int(sqrt))
        self.patterns = []
        self.state = None

        self.neurons = []
        n = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                neuron = Neuron(self, (i,j), n)
                self.neurons.append(neuron)
                n += 1
        self.set_random_state()
        self.weights = torch.zeros([self.n_neurons, self.n_neurons])
        #eye = np.diag_indices(self.weights.shape[0])   # set diagonal to 0
        #self.weights[eye[0], eye[1]] = torch.zeros(self.weights.shape[0])

        logger.info(f"Initialized Hopfield network of size {self.size} with {self.n_neurons} Neurons")

    def visualize_weight_matrix(self):
        plt.matshow(self.weights, cmap="binary")
        plt.show()

    def train(self):
        """
        Train the network to learn a pattern.

        :return: None
        """
        self.weights = torch.zeros([self.n_neurons, self.n_neurons])
        for neuron_i in self.neurons:
            for neuron_j in self.neurons:
                if neuron_i is neuron_j:
                    continue
                for pattern in self.patterns:
                    self.weights[neuron_i.n, neuron_j.n] += pattern[neuron_i.i, neuron_i.j] * pattern[neuron_j.i, neuron_j.j]
                self.weights[neuron_i.n, neuron_j.n] /= len(self.patterns)

    def add_pattern(self, pattern):
        """
        Add pattern to list of patterns.

        :param pattern: torch.Tensor of pattern
        :return: None
        """
        print(pattern)
        self.patterns.append(pattern)


    def run(self, steps):
        """
        Run the network for n steps.

        :param steps: int describing how often a neuron should be given the chance to update
        :return: None
        """
        for i in range(steps):
            neuron = np.random.choice(self.neurons)
            neuron.update()

    def visualize(self, state):
        """
        Return a matplotlib figure representing the binary state for each neuron.

        :param state: torch.Tensor describing binary state for each neuron
        :return: matplotlib.figure.Figure object
        """
        fig = Figure(figsize=(3, 3), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(state, cmap="Blues",  interpolation="nearest")  #TODO: Fix Bug where figure appears blanc when there is only one number in the tensor
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        return fig

    def get_random_state(self):
        """
        Get a random network state.

        :return: torch.Tensor
        """
        tensor = torch.rand((self.size[0], self.size[1])).round()   # generate random tensor of 0´s and 1´s
        tensor[tensor == 0] = -1  # replace 0 by -1
        return tensor

    def set_state(self, state):
        """
        Used to set a network state and set all neurons to reflect this state.

        :param state: torch.Tensor
        :return: None
        """
        self.state = state
        for neuron in self.neurons:
            neuron.state[0] = state[neuron.i, neuron.j]


    def set_state_from_neurons(self):
        """
        Sets Network state from neuron states.

        :return: None
        """
        for neuron in self.neurons:
            self.state[neuron.i, neuron.j] = neuron.state[0]

    def set_random_state(self):
        """
        Sets a random network state.

        :return: None
        """
        random_state = self.get_random_state()
        self.set_state(random_state)

    def visualize_pattern(self, i):
        """
        Returns a matplotlib figure representing the binary state for each neuron for a state saved in the network.

        :param i: index of pattern in self.patterns
        :return: matplotlib.figure.Figure object
        """
        return self.visualize(self.patterns[i])

    def create_pattern(self, char, size=None):
        """
        Get a pattern representing the binary states of each neuron in the network.

        :param char: str string to represent
        :param size: int size of the font, leave empty to auto fit
        :return: torch.Tensor
        """
        if size is None:
            size = self.size[0]
        # create font
        pil_font = ImageFont.truetype("arial.ttf", size=size // len(char), encoding="unic")
        text_width, text_height = pil_font.getsize(char)

        # create a blank canvas with extra space between lines
        canvas = Image.new("RGB", [size, size], (255, 255, 255))

        # draw the text onto the canvas
        draw = ImageDraw.Draw(canvas)
        offset = ((size - text_width) // 2,
                  (size - text_height) // 2)
        draw.text(offset, char, font=pil_font, fill="#000000")

        # Convert the canvas into an array with values in [0, 1]
        array = ((255 - np.asarray(canvas)) / 255.0)[:, :, 0].round()
        tensor = torch.from_numpy(array)
        tensor[tensor == 0] = -1  # replace 0 by -1
        return tensor

class Neuron(nn.Module):
    def __init__(self, network, position, n):
        super(Neuron, self).__init__()
        self.i = position[0]    # coordinates of the neuron in the network
        self.j = position[1]
        self.n = n

        self.network = network
        self.state = Tensor([0])
        self.bias = torch.Tensor([0])


    def activation_fn(self):
        total = 0
        bias_i = self.bias
        for neuron in self.network.neurons:
            if not neuron is self:
                state_j = neuron.state
                weight_ij = torch.unsqueeze(self.network.weights[self.n, neuron.n], 0)
                total += torch.matmul(weight_ij, state_j) + bias_i
        return total


    def update(self):
        if self.activation_fn() >= 0:
            self.state[0] = 1
        else:
            self.state[0] = -1

    def __repr__(self):
        return f"Neuron {self.i} {self.j} with state: {self.state[0]}"


if __name__ == '__main__':
    nn = HopfieldNetwork(25)
    nn.visualize_weight_matrix()

