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

    def _get_quiver_data(self, result, step=10):
        r_q = result.r[::step, :]
        G_q = result.G[::step, :]
        T_q = result.T[::step, :]
        return r_q, G_q, T_q
    
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

    def _plot_traj_3d_projection(self):
        
        self.fig_traj2d, self.ax_traj2d = plt.subplots(2,2, figsize=self.figsize)
        fig, ax = self.fig_traj2d, self.ax_traj2d
        
        for label, exp in self.experiments.items():
            res, color, qs = exp['result'], exp['color'], exp['quiver_scale']
            x, y, z = res.r[:,0], res.r[:,1], res.r[:,2]
            ax[0,0].plot(x, y, label='label') # (x,y)
            ax[0,1].plot(x, z) # (x,z)
            ax[1,0].plot(y, z) # (y,z)

            # Plotting gravity and thrust arrows
            r_q, G_q, T_q = self._get_quiver_data(res)
            ax[0,0].quiver(r_q[:,0], r_q[:,1], G_q[:,0], G_q[:,1], color=color, scale=self.quiver_scale, label=rf'Gravity/{qs}')
            ax[0,0].quiver(r_q[:,0], r_q[:,1], T_q[:,0], T_q[:,1], color='k',scale=self.quiver_scale, label=rf'Thrust/{qs}')
            ax[0,1].quiver(r_q[:,0], r_q[:,2], G_q[:,0], G_q[:,2], color=color, scale=self.quiver_scale, label=f'Gravity/{qs}')
            ax[0,1].quiver(r_q[:,0], r_q[:,2], T_q[:,0], T_q[:,2], color='k', scale=self.quiver_scale, label=f'Thrust/{qs}')
            ax[1,0].quiver(r_q[:,1], r_q[:,2], G_q[:,1], G_q[:,2], color=color, scale=self.quiver_scale, label=f'Gravity/{qs}')
            ax[1,0].quiver(r_q[:,1], r_q[:,2], T_q[:,1], T_q[:,2], color='k', scale=self.quiver_scale, label=f'Thrust/{qs}')

        # BC (x,y)
        ax[0,0].scatter(res.r0[:2],color='r',marker='o',label=r'$r(t=0)$')
        ax[0,0].scatter(res.rN[:2],color='r',marker='x',label=r'$r(t=1)$')
        ax[0,0].set_xlabel('x')
        ax[0,0].set_ylabel('y')
        # BC (x,z)
        ax[0,1].scatter(res.r0[0],res.r0[2],color='r',marker='o',label=r'$r(t=0)$')
        ax[0,1].scatter(res.rN[0],res.rN[2],color='r',marker='x',label=r'$r(t=1)$')
        ax[0,1].set_xlabel('x')
        ax[0,1].set_ylabel('z')
        # BC (y,z)
        ax[1,0].scatter(res.r0[1],res.r0[2],color='r',marker='o',label=r'$r(t=0)$')
        ax[1,0].scatter(res.rN[1],res.rN[2],color='r',marker='x',label=r'$r(t=1)$')
        ax[1,0].set_xlabel('y')
        ax[1,0].set_ylabel('z')

        # Legend
        self._plot_masses_3d(ax, res.ao, projection='2d')
        h, l = ax[0,0].get_legend_handles_labels()
        ax[1,1].legend(h, l, loc='center')
        ax[1,1].set_frame_on(False)
        ax[1,1].set_xticks([])
        ax[1,1].set_yticks([])

        plt.tight_layout()
        plt.show()
        fig.savefig(self._generate_fig_name+'traj2d.png')

    def plot_traj_2d(self):
        if self.dim == 3:
            return self._plot_traj_3d_projection()
        
        self.fig_traj2d, self.ax_traj2d = plt.subplots(figsize=self.figsize)
        fig, ax = self.fig_traj2d, self.ax_traj2d

        for label, exp in self.experiments.items():
            # Plot trajectory and start/end points
            result, color, linestyle, quiver_scale = exp['result'], exp['color'], exp['linestyle'], exp['quiver_scale']
            ax.plot(result.r[:, 0], result.r[:, 1], linestyle=linestyle, color=color, label=label)
            
            # Plotting gravity and thrust arrows
            r_q, G_q, T_q = self._get_quiver_data(result)
            ax.quiver(r_q[:, 0], r_q[:, 1], G_q[:, 0], G_q[:, 1], color=color, scale=quiver_scale, label=f'Gravity/{quiver_scale}')
            ax.quiver(r_q[:, 0], r_q[:, 1], T_q[:, 0], T_q[:, 1], color='k', scale=quiver_scale, label=f'Thrust/{quiver_scale}')

        ax.plot(exp['result'].r0[0], exp['result'].r0[1], 'o', color='red', label=r'$r(t=0)$')
        ax.plot(exp['result'].rN[0], exp['result'].rN[1], 'x', color='red', label=r'$r(t=1)$')
        self._plot_masses_2d(ax, exp['result'].ao)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self._generate_fig_name() + '_traj2d.png')
        plt.show()

    def _plot_masses_2d(self, ax, ao, planet_size=200):
        colors = ['#006400', '#228B22', '#6B8E23']
        for i, (x, y, m) in enumerate(ao):
            ax.scatter(x, y, s=m*planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={m}$')

    def _plot_masses_3d(self, ax, ao, projection=None, planet_size=200):
        colors = ['#006400', '#228B22', '#6B8E23']
        for i, (x, y, z, m) in enumerate(ao):
            if projection == '3d':
                ax.scatter(x, y, z, s=m*planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={m}$')
            elif projection == '2d':
                ax[0,0].scatter(ao[0], ao[1], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[3]}$')
                ax[0,1].scatter(ao[0], ao[2], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=fr'$GM_{i+1}={ao[3]}$')
                ax[1,0].scatter(ao[1], ao[2], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[3]}$')
               
    def plot_traj_3d(self, plot_quiver=True):
        self.fig_3d, self.ax_3d = plt.subplots(figsize=(6,6))
        fig, ax = self.fig_3d, self.ax_3d
        ax = fig.add_subplot(111, projection='3d')

        for label, exp in self.experiments.items():
            res = exp['result']
            r_q, G_q, T_q = self._get_quiver_data(res)
            self.ax3d.plot3D(res.r[:,0], res.r[:,1], res.r[:2], label=label, c='#1f77b4')
            if plot_quiver:
                self.ax3d.quiver(r_q[:,0], r_q[:,1], r_q[:,2], G_q[:,0], G_q[:,1], G_q[:,2], color=exp['color'], label=f'Gravity')
                self.ax3d.quiver(r_q[:,0], r_q[:,1], r_q[:,2], T_q[:,0], T_q[:,1], T_q[:,2], color='k', label=f'Thrust')
            self.ax3d.set_xlabel(r'$x$')
            self.ax3d.set_ylabel(r'$y$')
            self.ax3d.set_zlabel(r'$z$')

        ax.scatter(*res.r0, marker='o', color='red', label=r'$r(t=0)$')
        ax.scatter(*res.rN, marker='x', color='red', label=r'$r(t=1)$')

        self._plot_masses_3d(ax, res.ao, projection='3d')
        self.ax3d.legend(loc='upper center', ncol=3) 
        self.fig_3d.savefig('traj3d.png')

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
        
        self.plot_traj_2d()
        if self.dim == 3:
            self.plot_traj_3d()
        self.plot_loss()
        self.plot_thrust()
        self.plot_gravity()
# %%
