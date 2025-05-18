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

    def __init__(self, experiments, dim=None, figsize=(8,8)):
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
                    color=exp.get('color', None),
                    quiver_scale = exp.get('quiver_scale', 20)
                )
    
    def add_experiment(self, label, result, linestyle="-", color=None, quiver_scale=10):
        self.experiments[label] = {
                "result": result,
                "linestyle": linestyle,
                "color": color if color is not None else self.get_random_hex_color(),
                "quiver_scale": quiver_scale
            }

    def get_random_hex_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    

    def _plot_traj_3d_single(self):
        pass
    
    def _plot_loss_single(self):
        pass

    def _plot_magnitude_single(self):
        pass

    def _plot_traj_2d_single(self, ax, label, result, linestyle, color, quiver_scale=20):

        # Plot trajectory and start/end points
        ax.plot(result.r[:, 0], result.r[:, 1], linestyle=linestyle, color=color, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Plotting gravity and thrust arrows
        step = len(result.r) // 10
        r_q = result.r[::step, :]
        G_q = result.G[::step, :]
        T_q = result.T[::step, :]
        ax.quiver(r_q[:, 0], r_q[:, 1], G_q[:, 0], G_q[:, 1], color='k', scale=quiver_scale, label=f'Gravity/{quiver_scale}')
        ax.quiver(r_q[:, 0], r_q[:, 1], T_q[:, 0], T_q[:, 1], color='#FFA500', scale=quiver_scale, label=f'Thrust/{quiver_scale}')

    def plot_traj_2d(self):
        
        self.fig_traj2d, self.ax_traj2d = plt.subplots(figsize=self.figsize)
        fig, ax = self.fig_traj2d, self.ax_traj2d

        for label, exp in self.experiments.items():
            self._plot_traj_2d_single(ax, label, exp['result'], exp['linestyle'], exp['color'], exp['quiver_scale'])
        
        ax.plot(exp['result'].r0[0], exp['result'].r0[1], 'o', color='red', label=r'$r(t=0)$')
        ax.plot(exp['result'].rN[0], exp['result'].rN[1], 'x', color='red', label=r'$r(t=1)$')
        self._plot_masses_2d(ax, exp['result'].ao)
        ax.set_aspect('equal')
        ax.legend(loc='best')
        fig.tight_layout()
        exp_str = "_".join([f"{k}" for k, _ in self.experiments.items()]) + 'traj2d.png'
        fig.savefig(exp_str)
        plt.show()


    def _plot_masses_2d(self, ax, ao, planet_size=200):
        colors = ['#006400', '#228B22', '#6B8E23']
        for i, (x, y, m) in enumerate(ao):
            ax.scatter(x, y, s=m*planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={m}$')
               

    def plot_traj_3d(self):
        pass

    def plot_loss(self):
        pass

    def plot_magnitude(self):
        pass
    
    def plot_all(self):
        pass
# %%
