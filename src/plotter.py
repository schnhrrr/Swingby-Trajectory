#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import random
from src.result import TrajectoryResult

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.unicode_minus": True,
    "font.size":11
})

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
    
    def _generate_fig_name(self):
        return "_".join([f"{k}" for k, _ in self.experiments.items()]) 

    def _plot_traj_3d_single(self):
        pass

    def plot_thrust(self):
        self.fig_thrust, self.ax_thrust = plt.subplots(figsize=self.figsize)
        fig, ax = self.fig_thrust, self.ax_thrust

        for label, exp in self.experiments.items():
            result = exp['result']
            ax.plot(result.t, result.T_mag, linestyle='solid', color=exp['color'], label=label)
            ax.set_xlabel('Normalized time')
            ax.set_ylabel('Thrust magnitude')
            ax.set_xlim(0, 1)

        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self._generate_fig_name() + '_thrust.png')
        plt.show()

    def plot_gravity(self):
        self.fig_gravity, self.ax_gravity = plt.subplots(figsize=self.figsize)
        fig, ax = self.fig_gravity, self.ax_gravity

        for label, exp in self.experiments.items():
            result = exp['result']
            ax.plot(result.t, result.a_mag, linestyle='solid', color=exp['color'], label=label+' Required Force Magnitude')
            ax.plot(result.t, result.G_mag, linestyle='dashed', color=exp['color'], label=label+' Gravity')
            ax.set_xlabel('Normalized time')
            ax.set_ylabel('Gravity / Force magnitude')
            ax.set_xlim(0, 1)

        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self._generate_fig_name() + '_gravity.png')
        plt.show()

    
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
        fig.savefig(self._generate_fig_name() + '_traj2d.png')
        plt.show()

    def _plot_masses_2d(self, ax, ao, planet_size=200):
        colors = ['#006400', '#228B22', '#6B8E23']
        for i, (x, y, m) in enumerate(ao):
            ax.scatter(x, y, s=m*planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={m}$')
               
    def plot_traj_3d(self):
        pass

    def plot_loss(self, x_lim=None):
        self.fig_loss, self.ax_loss = plt.subplots(figsize=self.figsize)
        fig, ax = self.fig_loss, self.ax_loss

        for label, exp in self.experiments.items():
            result = exp['result']
            ax.plot(result.loss, linestyle='solid', label=label+r' Total Loss')
            if result.loss_bc:
                ax.plot(result.loss_bc, linestyle='solid', label=label+r' $\omega_{BC}$$L_{BC}$')
            if result.loss_physics:
                ax.plot(result.loss_physics, linestyle='solid', label=label+r' $\omega_P$$L_{P}$')
            ax.set_xlabel('Training Epochs')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
        if x_lim:
            ax.set_xlim(0, x_lim) 
        else:
            max_len = max([len(exp['result'].loss) for exp in self.experiments.values()])
            ax.set_xlim(0, max_len)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self._generate_fig_name() + '_loss.png')
        plt.show()

    
    def plot_all(self):
        if self.dim == 2:
            self.plot_traj_2d()
        elif self.dim == 3:
            self.plot_traj_3d()
        self.plot_loss()
        self.plot_thrust()
        self.plot_gravity()
# %%
