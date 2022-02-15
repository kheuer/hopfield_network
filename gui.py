import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import tkinter as tk
from tkinter import Text, Label, Button, Frame, ttk, Entry, StringVar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from network import HopfieldNetwork
from functools import partial
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

default_n_neurons = 15 ** 2


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.init_network(default_n_neurons)
        self.pattern_index = 0
        self.control_menu = Frame(self.root)
        self.control_menu.grid(row=0, column=0)

        sep1 = ttk.Separator(self.root, orient="vertical")
        sep1.grid(row=0, column=1)
        self.current_menu = Frame(self.root)
        self.current_menu.grid(row=0, column=2)
        sep2 = ttk.Separator(self.root, orient="vertical")
        sep2.grid(row=0, column=3)

        self.pattern_menu = Frame(self.root)
        self.pattern_menu.grid(row=0, column=4)

        # control
        init_network_frame = Frame(self.control_menu)
        init_network_frame.pack()
        neuron_frame = Frame(init_network_frame)
        neuron_frame.grid(row=0, column=1)
        n_neuron_field = Entry(neuron_frame)
        n_neuron_field.insert(-1, default_n_neurons)
        n_neuron_field.pack()
        n_neurons_label = Label(neuron_frame, text="Number of Neurons")
        n_neurons_label.pack()
        init_network_btn = Button(init_network_frame, text="Reset Network", width=35, height=3,
                                  command=partial(self.network_action, self.init_network,
                                                  partial(self.get_numeric_input, n_neuron_field)))
        init_network_btn.grid(row=0, column=0)

        pattern_frame = Frame(self.control_menu)
        pattern_frame.pack()
        save_random_pattern = Button(pattern_frame, text="Save random pattern", width=55, height=3,
                                     command=partial(self.network_action, self.save_pattern, "random"))
        save_random_pattern.pack()

        chr_save_frame = Frame(pattern_frame)
        chr_save_frame.pack()
        ch_selection_frame = Frame(chr_save_frame)
        ch_selection_frame.grid(row=0, column=1)
        self.ch_selection = Entry(ch_selection_frame)
        self.ch_selection.insert(-1, "A")
        self.ch_selection.pack()
        ch_selection_label = Label(ch_selection_frame, text="Insert Character")
        ch_selection_label.pack()
        train_pattern_btn = Button(chr_save_frame, text="Save Character to Network", width=35, height=3,
                                   command=partial(self.network_action, partial(self.save_pattern, "character")))
        train_pattern_btn.grid(row=0, column=0)

        models_frame = Frame(pattern_frame)
        models_frame.pack()
        save_model1_btn = Button(models_frame, text="Load model ABC", width=27, height=3,
                                 command=partial(self.network_action, self.load_model, 1))
        save_model1_btn.grid(row=0, column=0)
        save_model2_btn = Button(models_frame, text="Load model ABCCCC", width=27, height=3,
                                 command=partial(self.network_action, self.load_model, 2))
        save_model2_btn.grid(row=0, column=1)
        save_model3_btn = Button(models_frame, text="Load model alphabet", width=27, height=3,
                                 command=partial(self.network_action, self.load_model, 3))
        save_model3_btn.grid(row=1, column=0)

        # current
        current_state_label = Label(self.current_menu, text="Current Network state")
        current_state_label.grid(row=0, column=0)
        placeholder_fig_state = Figure(figsize=(3, 3), dpi=100)
        current_plot = self.get_plot_widget(placeholder_fig_state, self.current_menu)
        current_plot.grid(row=1, column=0)

        current_menu_lower = Frame(self.current_menu)
        current_menu_lower.grid(row=2, column=0)
        random_state = Button(current_menu_lower, text="Randomise Network state", width=20, height=3,
                              command=partial(self.network_action, self.change_network_state, "random"))
        random_state.grid(row=0, column=0)
        clear_state = Button(current_menu_lower, text="Clear Network state", width=20, height=3,
                             command=partial(self.network_action, self.change_network_state, "empty"))
        clear_state.grid(row=0, column=1)

        # patterns
        self.pattern_desc = StringVar()
        self.pattern_desc.set("Saved patterns")
        pattern_label = Label(self.pattern_menu, textvariable=self.pattern_desc)
        pattern_label.grid(row=0, column=0, columnspan=2)

        # plot = self.get_plot_widget(placeholder_fig_pattern, self.pattern_menu)
        # plot.grid(row=1, column=0, columnspan=2)

        patterns_menu_lower = Frame(self.pattern_menu)
        patterns_menu_lower.grid(row=2, column=0)
        random_state = Button(patterns_menu_lower, text="Previous", width=20, height=3,
                              command=partial(self.network_action, self.change_pattern, -1))
        random_state.grid(row=0, column=0)
        clear_state = Button(patterns_menu_lower, text="Next", width=20, height=3,
                             command=partial(self.network_action, self.change_pattern, 1))
        clear_state.grid(row=0, column=1)

        self.network_action(lambda: None)
        self.root.mainloop()

    def get_numeric_input(self, field):
        val = field.get()
        try:
            return int(val)
        except ValueError:
            raise ValueError(f"You must provide an integer value.")

    def get_character_input(self, field):
        val = field.get()
        if len(val) == 1:
            if val.isalpha():
                return val
            else:
                raise ValueError("You must provide an alphabetical character")
        else:
            raise ValueError("You must provide a single character.")

    def change_pattern(self, change):
        patterns = self.network.patterns
        if not patterns:
            placeholder_fig_pattern = Figure(figsize=(3, 3), dpi=100)
            no_pattern_plot = placeholder_fig_pattern.add_subplot(111)
            no_pattern_plot.text(0.1, 0.5, "No Patterns are saved.", bbox=dict(facecolor="red", alpha=0.5), fontsize=12)
            placeholder_fig_pattern.axes[0].get_xaxis().set_visible(False)
            placeholder_fig_pattern.axes[0].get_yaxis().set_visible(False)
            plot = self.get_plot_widget(placeholder_fig_pattern, self.pattern_menu)
            plot.grid(row=1, column=0, columnspan=2)
            self.pattern_desc.set("Saved patterns")
            logger.debug("No patterns are saved.")
            return
        max_pattern_index = len(patterns) - 1
        self.pattern_index += change
        if self.pattern_index > max_pattern_index:
            self.pattern_index = 0
        elif self.pattern_index < 0:
            self.pattern_index = max_pattern_index
        pattern = self.network.patterns[self.pattern_index]
        fig = self.network.visualize(pattern)

        pattern_plot = self.get_plot_widget(fig, self.pattern_menu)
        pattern_plot.grid(row=1, column=0, columnspan=2)

        self.pattern_desc.set(f"Saved patterns (showing {self.pattern_index + 1} / {max_pattern_index + 1})")

    def translate_state(self, state):
        if isinstance(state, str):
            if state == "random":
                state = self.network.get_random_state()
            elif state == "empty":
                state = np.zeros(self.network.size)
            elif state == "character":
                state = self.network.create_pattern(self.get_character_input(self.ch_selection))
            elif len(state) == 1:   # is string of a single character that should be represented
                state = self.network.create_pattern(state)
        return state

    def change_network_state(self, state):
        state = self.translate_state(state)
        self.network.set_state(state)

    def save_pattern(self, pattern):
        state = self.translate_state(pattern)
        self.network.train(state)

    def load_model(self, number):
        if number == 1:
            logger.debug("Loaded Model 'ABC'")
            for ch in list("ABC"):
                self.save_pattern(ch)
        elif number == 2:
            logger.debug("Loaded Model 'ABCCCC'")
            for ch in list("ABCCCC"):
                self.save_pattern(ch)
        elif number == 3:
            logger.debug("Loaded Model 'Alphabet'")
            for ch in list("abcdefghijklmnopqrstuvwxyzäöü"):
                self.save_pattern(ch)



    def network_action(self, action, *args, **kwargs):
        logger.debug(f"Network action call to: {action} with args: {args} and kwargs: {kwargs}")
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, partial):
                args[i] = arg()
                logger.debug(f"Changed **args to {args}")
        args = tuple(args)
        for arg in kwargs:
            if isinstance(arg, partial):
                args = list(kwargs)
                args[kwargs.index(arg)] = arg()
                args = tuple(kwargs)
                logger.debug(f"Changed **kwargs to {kwargs}")

        action(*args, **kwargs)
        self.visualize_network()
        self.change_pattern(0)

    def init_network(self, n_neurons):
        self.pattern_index = 0
        self.network = HopfieldNetwork(n_neurons)

    def get_plot_widget(self, fig, master):
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        return canvas.get_tk_widget()

    def visualize_network(self):
        fig = self.network.visualize(self.network.state)
        current_plot = self.get_plot_widget(fig, self.current_menu)
        current_plot.grid(row=1, column=0, columnspan=2)

    def visualize_pattern(self, pattern):
        fig = self.network.visualize(pattern)
        self.add_plot(fig, 2, 1)
