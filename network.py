import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
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

    def create_pattern(self, char, size=None):
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
        return ((255 - np.asarray(canvas)) / 255.0)[:, :, 0].round()

if __name__ == '__main__':
    nn = HopfieldNetwork(100)
    
