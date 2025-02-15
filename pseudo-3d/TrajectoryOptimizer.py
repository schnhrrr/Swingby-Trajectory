#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.unicode_minus": True,
    "font.size":11
})

class PINN(nn.Module):

    def __init__(self, N_INPUT, N_OUTPUT, N_NEURONS, N_LAYERS):
        super().__init__()
        self.activation = nn.Tanh
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_NEURONS), self.activation()) # fully connected input layer
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_NEURONS, N_NEURONS),self.activation()]) for _ in range(N_LAYERS-1)])
        self.fco = nn.Linear(N_NEURONS, N_OUTPUT)
 
    def forward(self, t):
        firstlayer = self.fci(t)
        hiddenlayer = self.fch(firstlayer)
        xyz = self.fco(hiddenlayer)
        g = torch.cat([2*t-1, 2*t-1, torch.zeros_like(t)], dim=1)
        phi = torch.cat([t*(1-t),t*(1-t),t*(1-t)], dim=1)
        xyz_transformed = g + phi * xyz
        return xyz_transformed

class TrajectoryOptimizer:

    def __init__(self, model, ao_xyzgm, t_colloc, t_total, r0, r1, opt_lbfgs, opt_adam, n_lbfgs=150, n_adam=0, w_physics=1, quiver_scale=10, planet_size=200):
        self.model = model  # Pinn
        self.eps = 1e-8
        self.lbfgs = opt_lbfgs
        self.adam = opt_adam
        self.n_lbfgs = n_lbfgs
        self.n_adam = n_adam
        self.loss_history = []

        self.w_physics = w_physics

        self.t = t_colloc  # Collocation points
        self.t_total = t_total  # Time to go from r0 to r1
        self.r0 = r0  # r(t=0)
        self.r1 = r1  # r(t=1)
        self.ao_xyzgm = torch.tensor(ao_xyzgm)  # Astronomic objects (r, Gm)

        self.train_model()

        # Arrays to plot
        self.t_plt = self.t.detach().numpy().squeeze()
        self.r_plt = self.r.detach().numpy()
        self.v_plt = self.v.detach().numpy()
        self.a_plt = self.a.detach().numpy()
        self.G_plt = self.G.detach().numpy()
        self.T_plt = self.T.detach().numpy()
        self.ao_plt = ao_xyzgm
        self.quiver_scale = quiver_scale
        self.planet_size = planet_size

        # Magnitudes 
        self.a_mag = self.a.norm(dim=1).detach().numpy()
        self.G_mag = self.G.norm(dim=1).detach().numpy()
        self.T_mag = self.T.norm(dim=1).detach().numpy()
        
        self.plot_trajectory2d()
        self.plot_trajectory3d()
        self.plot_force_magnitudes()
        self.plot_loss()
    
    def compute_gravitational_force(self):

        self.G = torch.zeros_like(self.r)
        for ao in self.ao_xyzgm:
            r_diff = self.r - ao[:3]
            denominator = (torch.linalg.norm(r_diff, dim=1) + self.eps)**3
            self.G[:,0] += -ao[-1] * r_diff[:,0] / denominator
            self.G[:,1] += -ao[-1] * r_diff[:,1] / denominator
            self.G[:,2] += -ao[-1] * r_diff[:,2] / denominator

    def compute_thrust(self):

            x, y, z    = self.r[:,0], self.r[:,1], self.r[:,2]
            vx = torch.autograd.grad(x, self.t, create_graph=True, grad_outputs=torch.ones_like(x))[0] / self.t_total
            vy = torch.autograd.grad(y, self.t, create_graph=True, grad_outputs=torch.ones_like(y))[0] / self.t_total
            vz = torch.autograd.grad(z, self.t, create_graph=True, grad_outputs=torch.ones_like(z))[0] / self.t_total
            ax = torch.autograd.grad(vx, self.t, create_graph=True, grad_outputs=torch.ones_like(vx))[0] / self.t_total
            ay = torch.autograd.grad(vy, self.t, create_graph=True, grad_outputs=torch.ones_like(vy))[0] / self.t_total
            az = torch.autograd.grad(vz, self.t, create_graph=True, grad_outputs=torch.ones_like(vz))[0] / self.t_total

            self.v = torch.concat([vx,vy,vz],dim=1)
            self.a = torch.concat([ax,ay,az],dim=1)

            self.compute_gravitational_force()
            self.T = self.a - self.G 
        

    def train_model(self):

        for i in range(self.n_adam + self.n_lbfgs):
            if i % 20 == 0 and i < self.n_adam:
                print(f'ADAM: {i}')
            elif i % 10 == 0 and i > self.n_adam:
                print(f'LBFGS: {i}')
            if i < self.n_adam:
                self.optimizer = self.adam
                self.optimizer.zero_grad()
                self.closure()
                self.optimizer.step()
            else:
                self.optimizer = self.lbfgs
                self.optimizer.step(self.closure) 


    def closure(self):

        self.optimizer.zero_grad()
        self.r = self.model(self.t)
        self.compute_thrust()
        self.loss_physics = torch.mean(self.T.norm(dim=1)) * self.w_physics
        self.loss_history.append(self.loss_physics.detach())
        self.loss_physics.backward()

        return self.loss_physics

    def plot_loss(self):

        self.fig_loss = plt.figure()
        plt.plot(self.loss_history, label='Loss(Training epochs)')
        plt.yscale('log')
        plt.xlim(left=0)
        plt.xlim(right=len(self.loss_history))
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        self.fig_loss.savefig('loss.png')

    def plot_trajectory2d(self):

        self.x, self.y, self.z = self.r_plt[:,0], self.r_plt[:,1], self.r_plt[:,2]
        self.fig_traj, self.ax_traj = plt.subplots(2,2, figsize=(9,8))
        
        # (x,y)
        self.ax_traj[0,0].scatter(-1,-1,color='r',marker='o',label=r'$r(t=0)=(-1,-1,0)$')
        self.ax_traj[0,0].scatter(1,1,color='r',marker='x',label=r'$r(t=1)=(1,1,0)$')
        self.ax_traj[0,0].plot(self.x, self.y, label='Trajectory')
        self.ax_traj[0,0].set_xlabel('x')
        self.ax_traj[0,0].set_ylabel('y')

        # (x,z)
        self.ax_traj[0,1].scatter(-1,0,color='r',marker='o',label=f'r(t=0)=(-1,-1,-1)')
        self.ax_traj[0,1].scatter(1,0,color='r',marker='x',label=f'r(t=0)=(1,1,1)')
        self.ax_traj[0,1].plot(self.x, self.z)
        self.ax_traj[0,1].set_xlabel('x')
        self.ax_traj[0,1].set_ylabel('z')

        # (y,z)
        self.ax_traj[1,0].scatter(-1,0,color='r',marker='o',label=f'r(t=0)=(-1,-1,-1)')
        self.ax_traj[1,0].scatter(1,0,color='r',marker='x',label=f'r(t=0)=(1,1,1)')
        self.ax_traj[1,0].plot(self.y, self.z)
        self.ax_traj[1,0].set_xlabel('y')
        self.ax_traj[1,0].set_ylabel('z')

        # Plotting gravity and thrust arrows
        self.step = len(self.r_plt) // 10
        self.r_q = self.r_plt[::self.step,:]
        self.G_q = self.G_plt[::self.step,:]
        self.T_q = self.T_plt[::self.step,:]
        self.ax_traj[0,0].quiver(self.r_q[:,0], self.r_q[:,1], self.G_q[:,0], self.G_q[:,1], color='k', scale=self.quiver_scale, label=rf'Gravity/{self.quiver_scale}')
        self.ax_traj[0,0].quiver(self.r_q[:,0], self.r_q[:,1], self.T_q[:,0], self.T_q[:,1], color='#FFA500',scale=self.quiver_scale, label=rf'Thrust/{self.quiver_scale}')
        self.ax_traj[0,1].quiver(self.r_q[:,0], self.r_q[:,2], self.G_q[:,0], self.G_q[:,2], color='k', scale=self.quiver_scale, label='Gravity')
        self.ax_traj[0,1].quiver(self.r_q[:,0], self.r_q[:,2], self.T_q[:,0], self.T_q[:,2], color='#FFA500', scale=self.quiver_scale, label='Thrust')
        self.ax_traj[1,0].quiver(self.r_q[:,1], self.r_q[:,2], self.G_q[:,1], self.G_q[:,2], color='k', scale=self.quiver_scale, label='Gravity')
        self.ax_traj[1,0].quiver(self.r_q[:,1], self.r_q[:,2], self.T_q[:,1], self.T_q[:,2], color='#FFA500', scale=self.quiver_scale, label='Thrust')

        # Legend
        self.plot_masses(2)
        h, l = self.ax_traj[0,0].get_legend_handles_labels()
        self.ax_traj[1,1].legend(h, l, loc='center')
        self.ax_traj[1,1].set_frame_on(False)
        self.ax_traj[1,1].set_xticks([])
        self.ax_traj[1,1].set_yticks([])

        plt.tight_layout()
        plt.show()
        self.fig_traj.savefig('traj.png')

    def plot_trajectory3d(self):

        self.fig_3d = plt.figure(figsize=(6,6))
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax3d.scatter(-1,-1,0, marker='o', color='r', label=f'r(0)=(-1,-1,0)')
        self.ax3d.scatter(1,1,0, marker='x', color='r', label=f'r(1)=(1,1,0)')  
        self.ax3d.plot3D(self.x, self.y, self.z, label='Trajectory', c='#1f77b4')
        self.ax3d.quiver(self.r_q[:,0], self.r_q[:,1], self.r_q[:,2], self.G_q[:,0], self.G_q[:,1], self.G_q[:,2], color='k', label=f'Gravity')
        self.ax3d.quiver(self.r_q[:,0], self.r_q[:,1], self.r_q[:,2], self.T_q[:,0], self.T_q[:,1], self.T_q[:,2], color='#FFA500', label=f'Thrust')
        self.ax3d.set_xlabel(r'$x$')
        self.ax3d.set_ylabel(r'$y$')
        self.ax3d.set_zlabel(r'$z$')
        
        self.plot_masses(3)
        self.ax3d.legend(loc='upper center', ncol=3) 
        self.fig_3d.savefig('traj3d.png')
        

    def plot_masses(self, dim):

        colors = ['#006400', '#228B22', '#6B8E23']
        for i, ao in enumerate(self.ao_plt):
            if dim == 2:
                self.ax_traj[0,0].scatter(ao[0], ao[1], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[3]}$')
                self.ax_traj[0,1].scatter(ao[0], ao[2], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=fr'$GM_{i+1}={ao[3]}$')
                self.ax_traj[1,0].scatter(ao[1], ao[2], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[3]}$')
            if dim == 3:
                self.ax3d.scatter(*ao[:3], s=ao[3]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[3]}$')
   
    def plot_force_magnitudes(self):
        
        self.fig_forces = plt.figure()
        plt.plot(self.t_plt, self.T_mag + self.G_mag, label='Required force magnitude')
        plt.plot(self.t_plt, self.G_mag, label='Total gravity', color='k', linestyle='--')
        plt.plot(self.t_plt, self.T_mag, label='Thrust magnitude', color='g')
        plt.fill_between(self.t_plt, np.zeros_like(self.t_plt), self.T_mag, color='g')
        plt.xlabel('Normalized time')
        plt.ylabel('Exerted force')
        plt.xlim(0,1)
        plt.legend()
        plt.show()
        plt.savefig('force.png')

# %%

t_colloc = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)
t_total = torch.tensor(1.0, requires_grad=True)

r0 = torch.tensor([[-1.,-1.,0.]])
r1 = torch.tensor([[1.,1.,0.]])

ao_xyzgm = [[-0.5, -1., 0., 0.5],  # astronomic objects: x, y, z gravitational mass
            [-0.2, 0.4, 0., 1.0],
            [ 0.8, 0.3, 0., 0.5]]
   
# Initialize model
pinn = PINN(1, 3, 50, 3)
params = list(pinn.parameters())
params.append(t_total)
optimizer_adam = torch.optim.Adam(params, lr=1e-4)
optimizer_lbfgs = torch.optim.LBFGS(params, lr=.1, max_iter=10)

# Set random seed
seed = 123
torch.manual_seed(seed)

obj = TrajectoryOptimizer(pinn, ao_xyzgm, t_colloc, t_total, r0, r1, optimizer_lbfgs, optimizer_adam, n_adam=0, w_physics=1)

# %%
