import matplotlib.pyplot as plt
import numpy as np
from result import TrajectoryResult
import random

class TrajectoryPlotter:

    def __init__(self, dim=None, figsize=(8,6), **experiments):
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
        
        #TODO: decide if we want to use a dict or a list
        for exp in experiments:
            label = exp['result'].label
            self.experiments[label]['result'] = exp['result']
            self.experiments[label]['linestyle'] = exp.get('linestyle', '-')
            self.experiments[label]['color'] = exp.get('color', self.get_random_hex_color())
            
    def get_random_hex_color():
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