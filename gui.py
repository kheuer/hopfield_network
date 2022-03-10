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
import time



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

        chr_save_frame = Frame(pattern_frame)
        chr_save_frame.pack()
        ch_selection_frame = Frame(chr_save_frame)
        ch_selection_frame.grid(row=0, column=1)
        self.ch_selection = Entry(ch_selection_frame)
        self.ch_selection.insert(-1, "1")
        self.ch_selection.pack()
        ch_selection_label = Label(ch_selection_frame, text="Insert Number")
        ch_selection_label.pack()
        train_pattern_btn = Button(chr_save_frame, text="Save Number to Network", width=35, height=3,
                                   command=partial(self.network_action, partial(self.save_pattern, "number")))
        train_pattern_btn.grid(row=0, column=0)

        models_frame = Frame(pattern_frame)
        models_frame.pack()
        save_model1_btn = Button(models_frame, text="Load model 1/2", width=27, height=3,
                                 command=partial(self.network_action, self.load_model, "1/2"))
        save_model1_btn.grid(row=0, column=0)
        save_model2_btn = Button(models_frame, text="Load model 1/5/8", width=27, height=3,
                                 command=partial(self.network_action, self.load_model, "1/5/8"))
        save_model2_btn.grid(row=0, column=1)

        save_n_random_patterns_frame = Frame(pattern_frame)
        save_n_random_patterns_frame.pack()
        n_patterns_frame = Frame(save_n_random_patterns_frame)
        n_patterns_frame.grid(row=0, column=1)
        self.n_patterns_field = Entry(n_patterns_frame)
        self.n_patterns_field.insert(-1, 1)
        self.n_patterns_field.pack()
        n_patterns_label = Label(n_patterns_frame, text="Number of Patterns")
        n_patterns_label.pack()
        save_random_pattern_btn = Button(save_n_random_patterns_frame, text="Save random pattern", width=35, height=3,
                                     command=partial(self.network_action, self.load_model, "n_random"))
        save_random_pattern_btn.grid(row=0, column=0)



        advance_network_frame = Frame(self.control_menu)
        advance_network_frame.pack()
        advance_network_1_btn = Button(advance_network_frame, text="Advance 100 Steps", width=27, height=3,
                                 command=partial(self.network_action, self.advance_model, 100))
        advance_network_1_btn.grid(row=0, column=0)
        advance_network_10_btn = Button(advance_network_frame, text="Solve model", width=27, height=3,
                                       command=partial(self.network_action, self.solve_network))
        advance_network_10_btn.grid(row=0, column=1)

        advance_network_n_frame = Frame(self.control_menu)
        advance_network_n_frame.pack()
        n_steps_frame = Frame(advance_network_n_frame)
        n_steps_frame.grid(row=0, column=1)
        n_steps_field = Entry(n_steps_frame)
        n_steps_field.insert(-1, self.network.n_neurons)
        n_steps_field.pack()
        n_steps_label = Label(n_steps_frame, text="Number of Steps")
        n_steps_label.pack()
        advance_network_n_btn = Button(advance_network_n_frame, text="Advance Network n steps", width=35, height=3,
                                  command=partial(self.network_action, self.advance_model, partial(self.get_numeric_input, n_steps_field)))
        advance_network_n_btn.grid(row=0, column=0)



        # current
        self.state_desc = StringVar()
        self.set_network_desc()
        current_state_label = Label(self.current_menu, textvariable=self.state_desc)
        current_state_label.grid(row=0, column=0)
        placeholder_fig_state = Figure(figsize=(3, 3), dpi=100)
        current_plot = self.get_plot_widget(placeholder_fig_state, self.current_menu)
        current_plot.grid(row=1, column=0)

        current_menu_lower = Frame(self.current_menu)
        current_menu_lower.grid(row=2, column=0)
        random_state = Button(current_menu_lower, text="Randomise Network\nstate", width=13, height=3,
                              command=partial(self.network_action, self.change_network_state, "random"))
        random_state.grid(row=0, column=0)
        copy_state = Button(current_menu_lower, text="Set shown state", width=13, height=3,
                             command=partial(self.network_action, self.change_network_state, "current"))
        copy_state.grid(row=0, column=1)
        mutate_state = Button(current_menu_lower, text="Mutate state", width=13, height=3,
                            command=partial(self.network_action, self.change_network_state, "mutate"))
        mutate_state.grid(row=0, column=2)

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


        plot_weights = Button(self.control_menu, text="Plot Weights", width=20, height=3,   # Debug
                             command=self.visualize_weight_matrix)
        plot_weights.pack()



        self.network_action(lambda: None)
        self.root.mainloop()

    def set_network_desc(self):
        is_stable = self.network.is_in_local_minima()
        stable_desc = {True: "Stable", False: "Unstable"}[is_stable]
        is_saved = self.network.pattern_is_saved(self.network.state)
        saved_desc = {True: "Saved", False: "not Saved"}[is_saved]
        if is_saved:
            index = self.network.get_pattern_index(self.network.state)
            pattern_number = index + 1
        else:
            pattern_number = "n/a"

        self.state_desc.set(f"Current Network state ({stable_desc}) \nPattern is {saved_desc} \nPattern Number: {pattern_number}")


    def get_numeric_input(self, field):
        val = field.get()
        try:
            return int(val)
        except ValueError:
            raise ValueError(f"You must provide an integer value.")

    def get_character_input(self, field):
        val = field.get()
        if len(val) == 1:
            return val
        elif len(val) > 1:
            logger.info(f"String must be a single character, shortened to: {val[0]}")
            return val[0]
        else:
            logger.info("You must provide a value, using: '.'")
            return "."

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
            elif state == "number":
                state = self.network.create_pattern(self.get_numeric_input(self.ch_selection))
            elif state == "current":
                if not self.network.patterns:
                    logger.warning("You must train at least one state first.")
                    state = self.translate_state("empty")
                else:
                    state = self.network.patterns[self.pattern_index]
            elif state == "mutate":
                current = self.network.state.clone()
                for i, row in enumerate(current):
                    for j, val in enumerate(row):
                        if np.random.random() < 0.05:
                            current[i, j] = 0 - current[i, j]
                state = current

            elif len(state) == 1:   # is string of a single number that should be represented
                state = self.network.create_pattern(int(state))

        return state.clone()    # return deep copy to avoid unexpected side effects

    def advance_model(self, steps):
        logger.info(f"Advance Network by {steps} steps.")
        self.network.run(steps)
        self.network.set_state_from_neurons()

    def change_network_state(self, state):
        state = self.translate_state(state)
        self.network.set_state(state)

    def save_pattern(self, pattern):
        state = self.translate_state(pattern)
        self.network.add_pattern(state)
        if pattern in ["number"]:
            self.network.train()

    def load_model(self, model_desc):
        if model_desc == "1/2":
            for ch in ["1", "2"]:
                self.save_pattern(ch)
            logger.debug("Loaded Model '1/2'")
        elif model_desc == "1/5/8":
            for ch in ["1", "5", "8"]:
                self.save_pattern(ch)
            logger.debug("Loaded Model '1/5/8'")
        elif model_desc == "n_random":
            for i in range(self.get_numeric_input(self.n_patterns_field)):
                self.save_pattern("random")
        self.network.train()

    def visualize_weight_matrix(self):
        self.network.visualize_weight_matrix()

    def network_action(self, action, *args, **kwargs):
        logger.debug(f"Network action call to: {action} with args: {args} and kwargs: {kwargs}")
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, partial):
                args[i] = arg()
                #logger.debug(f"Changed **args to {args}")
        args = tuple(args)
        for arg in kwargs:
            if isinstance(arg, partial):
                args = list(kwargs)
                args[kwargs.index(arg)] = arg()
                args = tuple(kwargs)
                #logger.debug(f"Changed **kwargs to {kwargs}")

        action(*args, **kwargs)
        self.visualize_network()
        self.change_pattern(0)
        self.set_network_desc()

    def solve_network(self):
        self.network.solve()

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

if __name__ == '__main__':
    logger.info("starting.")
    gui = GUI()
    logger.info("exiting.")
