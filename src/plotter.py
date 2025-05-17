#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import random
from src.result import TrajectoryResult

# Developing
from config.config_2d import position2d_config, vanilla2d_config
from src.runner import run_experiment
results = []
for exp in [position2d_config, vanilla2d_config]:
    results.append(run_experiment(exp))

class TrajectoryPlotter:

    def __init__(self, experiments, dim=None, figsize=(8,6)):
        """
        Class to plot the results of the trajectory optimization.
        **experiments: keyword args where key is the name (str), and value is a dict:
            {
                'result': TrajectoryResult object (required),
                'linestyle': matplotlib linestyle (optional),
                'color': matplotlib color (optional)
            }
        """
        self.dim = dim
        self.figsize = figsize
        self.experiments = {}
        
        if experiments:
            for exp in experiments:
                self.add_experiment(
                    label=exp['label'],
                    result=exp['result'],
                    linestyle=exp.get('linestyle', '-'),
                    color=exp.get('color', None)
                )
    
    def add_experiment(self, label, result, linestyle="-", color=None):
        self.experiments[label] = {
                "result": result,
                "linestyle": linestyle,
                "color": color if color is not None else self.get_random_hex_color()
            }

    def get_random_hex_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    def _plot_traj_2d_single(self):
        pass

    def _plot_traj_3d_single(self):
        pass
    
    def _plot_loss_single(self):
        pass

    def _plot_magnitude_single(self):
        pass

    def plot_traj_2d(self):
        pass

    def plot_traj_3d(self):
        pass

    def plot_loss(self):
        pass

    def plot_magnitude(self):
        pass
    
    def plot_all(self):
        pass
# %%
