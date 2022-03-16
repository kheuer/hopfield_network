import torch
import torchvision
from torchvision.datasets import USPS
from torch import Tensor, nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageDraw, ImageFont
import time
import os

import logging
logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dataset = torchvision.datasets.MNIST(os.getcwd() + "/files/MNIST/", train=True, download=True)

class HopfieldNetwork:
    def __init__(self, n_neurons, updating_type="no_replace", testing=False):
        """
        Initialize a Hopfield Network with a quadratic shape of (n_neurons**0.5, n_neurons**0.5)

        :param testing: Boolean, True if the instance is created to perform statistical modeling, False if used with
                        Full functionality. If True, energy will not be calculated at update.
        :param updating_type: "replace" or "no_replace". Describes if updates should be performed with or
                                without replacement
        :param n_neurons: The number of neurons the network should have.
        :param testing: Boolean, True if using for testing/modeling, False if using for GUI
        """
        if updating_type == "replace":
            self.replace = True
        elif updating_type == "no_replace":
            self.replace = False
        else:
            raise ValueError("Unrecognized update type.")

        if n_neurons < 4:
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be at least 4")
        sqrt = np.sqrt(n_neurons)
        if not sqrt.is_integer():
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be divisible by itself to an int")
        if n_neurons < 100:
            logger.warning("We recommend to choose n_neurons to be >= 100 to ensure proper generation of characters.")
        self.testing = testing
        self.n_neurons = n_neurons
        self.size = (int(sqrt), int(sqrt))
        self.patterns = []
        self.energies = []
        self.state = None
        self.neurons = []
        self.weights = torch.zeros([self.n_neurons, self.n_neurons])
        n = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                neuron = Neuron(self, (i,j), n)
                self.neurons.append(neuron)
                n += 1
        self.set_random_state()

        self.weights = torch.zeros([self.n_neurons, self.n_neurons])
        if not self.testing:
            self.energy = self.get_energy()
        logger.info(f"Initialized Hopfield network of size {self.size} with {self.n_neurons} Neurons")

    def get_energy(self, pattern=None):
        pattern_desc = "Manual"
        if pattern is None:
            pattern = self.state
            pattern_desc = "State"
        energy = 0
        for neuron_i, neuron_j in IterNeurons(self.neurons):
            if neuron_i.state == neuron_j.state:
                energy += self.weights[neuron_i.n, neuron_j.n]
            else:
                energy -= self.weights[neuron_i.n, neuron_j.n]
        energy *= -1
        energy -= torch.sum(pattern)
        logger.debug(f"{pattern_desc} Pattern Energy: {energy}")
        return energy

    def visualize_weight_matrix(self):
        plt.matshow(self.weights, cmap="Blues")
        plt.show()

    def train(self):
        """
        Train the network to learn a pattern.

        :return: None
        """
        start = time.time()
        for neuron_i, neuron_j in IterNeurons(self.neurons):
            hebbian_sum = 0
            for pattern in self.patterns:
                hebbian_sum += pattern[neuron_i.i, neuron_i.j] * pattern[neuron_j.i, neuron_j.j]
            hebbian_weight = hebbian_sum / len(self.patterns)
            self.weights[neuron_i.n, neuron_j.n], self.weights[neuron_j.n, neuron_i.n] = hebbian_weight, hebbian_weight
        self.set_state_from_neurons()
        logger.debug(f"Finished training in {int(time.time()- start)} seconds.")
        if not self.testing:
            self.energy = self.get_energy()

    def add_pattern(self, pattern):
        """
        Add pattern to list of patterns.

        :param pattern: torch.Tensor of pattern
        :return: None
        """
        self.patterns.append(pattern.float())
        self.train()

    def run_with_replacement(self, steps):
        """
        Run the network for n steps.
        This is performed by choosing n neurons with replacement and updating them.

        :param steps: int describing how many neurons should be given the chance to update
        :return: None
        """
        start = time.time()
        for i in range(steps):
            neuron = np.random.choice(self.neurons)
            neuron.update()
        logger.debug(f"Finished network update with replacement in {int(time.time() - start)} seconds.")
        if not self.testing:
            self.energy = self.get_energy()

    def run_without_replacement(self, steps):
        """
        Run the network for n steps.
        This is performed by choosing n neurons without replacement and updating them.

        :param steps: int describing how many neurons should be given the chance to update
        :return: None
        """
        if steps < self.n_neurons:
            logger.warning(f"Please choose at least n_neurons={self.n_neurons} steps.")
        start = time.time()
        for i in sorted(np.arange(steps), key=lambda k: np.random.random()):
            while i >= self.n_neurons:
                logger.debug(f"Changed index {i} to {i-self.n_neurons}")
                i -= self.n_neurons
            self.neurons[i].update()
        logger.debug(f"Finished network update without replacement in {int(time.time() - start)} seconds.")
        if not self.testing:
            self.energy = self.get_energy()

    def run(self, steps):
        """
        Call run function according to neuron sampling type.

        :param steps: int describing how many neurons should be given the chance to update
        :return: None
        """
        if self.replace:
            self.run_with_replacement(steps)
        else:
            self.run_without_replacement(steps)


    def pattern_is_saved(self, pattern):
        """
        Determine whether a pattern is saved in the network.

        :param pattern: Pytorch Tensor symbolizing state.
        :return: Boolean, True if pattern is saved, False if not
        """
        for saved_pattern in self.patterns:
            if torch.equal(pattern, saved_pattern):
                return True
        return False

    def get_pattern_index(self, pattern):
        """

        :param pattern: Pytorch Tensor that is stored in the network.
        :return: int, index of the pattern
        """
        for i, saved_pattern in enumerate(self.patterns):
            if torch.equal(pattern.float(), saved_pattern.float()):
                return i
        raise ValueError("Pattern is not saved.")

    def make_n_changes(self, n):
        """
        Make n changes to the network. (Different to run because only updates are counted)

        :param n: N changes the network should undertake.
        :return: None
        :raise: RuntimeError if n_changes are not possible
        """
        if self.is_in_local_minima():
            self.set_state_from_neurons()
            logger.info("Network is already in local minima")
            return
        start = time.time()
        changes_made = 0
        while True:
            neuron = np.random.choice(self.neurons)
            if neuron.can_update():
                neuron.update()
                changes_made += 1
                if changes_made == n:
                    break
                elif self.is_in_local_minima():
                    self.set_state_from_neurons()
                    raise RuntimeError("Network is already in local minima.")
        self.set_state_from_neurons()
        logger.debug(f"Made {changes_made} updates in {int(time.time()-start)} seconds.")

    def solve(self):
        """
        Advance the network until it cannot advance any further.

        :return: None
        """
        logger.debug("Starting to solve network.")
        start = time.time()
        while True:
            self.run(self.n_neurons)
            if self.is_in_local_minima():
                break
        self.set_state_from_neurons()
        logger.debug(f"Solved network in {int(time.time() - start)} seconds.")

    def is_in_local_minima(self):
        """
        Determine wether the current state of the network can be left.

        :return: boolean, True if the network is in a minima, False if not
        """
        for neuron in self.neurons:
            if neuron.can_update():
                return False
        return True

    def visualize(self, state):
        """
        Return a matplotlib figure representing the binary state for each neuron.

        :param state: torch.Tensor describing binary state for each neuron
        :return: matplotlib.figure.Figure object
        """
        fig = Figure(figsize=(3, 3), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(state, cmap="Blues",  interpolation="nearest")
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
        self.state = state.float()
        for neuron in self.neurons:
            neuron.state[0] = state[neuron.i, neuron.j]



    def set_state_from_neurons(self):
        """
        Sets Network state from neuron states.

        :return: None
        """
        for neuron in self.neurons:
            self.state[neuron.i, neuron.j] = neuron.state[0]
        self.state = self.state.float()

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

    def create_pattern(self, number):
        """
        Get a pattern representing the binary states of each neuron in the network.

        :param number: int number to represent.
        :return: torch.Tensor
        """
        while True:
            img, n = dataset[np.random.randint(len(dataset))]
            if n == number:
                break
        img = img.resize(self.size)
        array = np.asarray(img)
        array = np.round(array / 255)
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


    def activation_fn(self):
        """
        Activation function of a McCulloch-Pitts-Neuron in the Hopfield Network.

        :return: Strength of input into Neuron
        """
        total = 0
        for neuron in self.network.neurons:
            if neuron is not self:
                if neuron.state == 1:
                    # if neuron state is one the product of weight and state will be the weight_ij
                    total += self.network.weights[self.n, neuron.n]
                else:
                    # if neuron state is one the product of weight and state will be the negative weight_ij
                    total -= self.network.weights[self.n, neuron.n]
        return total

    def update(self):
        """
        Update Neuron

        :return: None
        """
        if self.activation_fn() >= 0:
            self.state[0] = 1
        else:
            self.state[0] = -1

    def can_update(self):
        """
        Determine whether the neuron can update.

        :return: True if Neuron can update, False if not
        """
        if self.activation_fn() >= 0:
            proper_state = 1
        else:
            proper_state = -1
        if self.state[0] != proper_state:
            return True
        else:
            return False

class IterNeurons:
    """
    Helper class to iterate through all neurons in a list paired with every other one once.
    """
    def __init__(self, lst):
        self.lst = lst
        self.i = 0
        self.j = 0
        self.stop = len(lst) - 1
        self.shown = {}

    def __iter__(self):
        return self

    def __next__(self):
        i, j = self.i, self.j
        if i == self.stop and j == self.stop:
            raise StopIteration

        if j < self.stop:
            self.j += 1
        if j == self.stop:
            self.j = 0
            self.i += 1

        if (i, j) in self.shown or i == j:
            return self.__next__()
        else:
            self.shown[(j, i)] = None
            return self.lst[i], self.lst[j]

