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

    def __init__(self, N_INPUT, N_OUTPUT, N_NEURONS, N_LAYERS, phi, psi, R_cart, R_dot_cart, V_cart):
        super().__init__()
        self.activation = nn.Tanh
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_NEURONS), self.activation()) # fully connected input layer
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_NEURONS, N_NEURONS),self.activation()]) for _ in range(N_LAYERS-1)])
        self.fco = nn.Linear(N_NEURONS, N_OUTPUT)
        
        self.phi, self.psi = phi, psi
        self.R_cart, self.R_dot_cart, self.V_cart = R_cart, R_dot_cart, V_cart
    
    def forward(self, t, tN):
        firstlayer = self.fci(t)
        hiddenlayer = self.fch(firstlayer)
        x_y_model = self.fco(hiddenlayer)
        
        R_cart = self.R_cart(t, tN)
        R_dot_cart = self.R_dot_cart(t, tN)
        V_cart = self.V_cart(t, tN)

        return R_cart + self.psi(t, tN) * x_y_model + self.phi(t, tN)*(V_cart-R_dot_cart)

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
        
        self.plot_trajectory()
        self.plot_force_magnitudes()
        self.plot_loss()
    
    def compute_gravitational_force(self):

        self.G = torch.zeros_like(self.r)
        for ao in self.ao_xyzgm:
            r_diff = self.r - ao[:2]
            denominator = (torch.linalg.norm(r_diff, dim=1) + self.eps)**3
            self.G[:,0] -= ao[-1] * r_diff[:,0] / denominator
            self.G[:,1] -= ao[-1] * r_diff[:,1] / denominator

    def compute_thrust(self):

            x, y, = self.r[:,0], self.r[:,1]
            vx = torch.autograd.grad(x, self.t, create_graph=True, grad_outputs=torch.ones_like(x))[0] / self.t_total
            vy = torch.autograd.grad(y, self.t, create_graph=True, grad_outputs=torch.ones_like(y))[0] / self.t_total
            ax = torch.autograd.grad(vx, self.t, create_graph=True, grad_outputs=torch.ones_like(vx))[0] / self.t_total
            ay = torch.autograd.grad(vy, self.t, create_graph=True, grad_outputs=torch.ones_like(vy))[0] / self.t_total

            self.v = torch.concat([vx,vy],dim=1)
            self.a = torch.concat([ax,ay],dim=1)

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
        self.r = self.model(self.t, self.t_total)
        self.compute_thrust()
        self.loss_physics = torch.mean(self.T[10:90,:].norm(dim=1)) * self.w_physics
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

    def plot_trajectory(self):
        rho, theta = self.r_plt[:, 0], self.r_plt[:, 1]
        x, y = rho*np.cos(theta), rho*np.sin(theta)
        self.fig_traj, self.ax_traj = plt.subplots()
        
        # (x, y)
        self.ax_traj.plot(x, y, label='Trajectory')
        self.ax_traj.set_xlabel('x')
        self.ax_traj.set_ylabel('y')
        
        circle_leo = plt.Circle((0,0), R_earth + h_leo, color='b', fill=False, linestyle='dashed', label='Low earth orbit')
        circle_geo = plt.Circle((0,0), R_earth + h_geo, color='g', fill=False, linestyle='dashed', label='Geostationary orbit')
        self.ax_traj.add_patch(circle_leo)
        self.ax_traj.add_patch(circle_geo)
        # Plotting gravity and thrust arrows
        #step = len(self.r_plt) // 10
        #self.r_q = self.r_plt[::step, :]
        #self.G_q = self.G_plt[::step, :]
        #self.T_q = self.T_plt[::step, :]
        #self.ax_traj.quiver(self.r_q[:, 0], self.r_q[:, 1], self.G_q[:, 0], self.G_q[:, 1], color='k', scale=self.quiver_scale, label=rf'Gravity/{self.quiver_scale}')
        #self.ax_traj.quiver(self.r_q[:, 0], self.r_q[:, 1], self.T_q[:, 0], self.T_q[:, 1], color='#FFA500', scale=self.quiver_scale, label=rf'Thrust/{self.quiver_scale}')
        
        # Legend
        self.plot_masses()
        self.ax_traj.legend(loc='best')
        
        plt.show()
        self.fig_traj.savefig('traj.png')

    def plot_masses(self):
        colors = ['#006400', '#228B22', '#6B8E23']
        for i, ao in enumerate(self.ao_plt):
            self.ax_traj.scatter(ao[0], ao[1], color='k', label='Earth') #s=ao[2]*self.planet_size, color=colors[i], marker='o', label=f'$GM_{i+1}={ao[2]}$')
           
    def plot_force_magnitudes(self):
        
        self.fig_force = plt.figure()
        plt.plot(self.t_plt, self.T_mag + self.G_mag, label='Required force magn.')
        plt.plot(self.t_plt, self.G_mag, label='Total gravity', color='k', linestyle='--')
        plt.plot(self.t_plt, self.T_mag, label='Thrust magn.', color='g')
        plt.fill_between(self.t_plt, np.zeros_like(self.t_plt), self.T_mag, color='g')
        plt.xlabel('Normalized time')
        plt.ylabel('Exerted force')
        plt.xlim(0,1)
        plt.legend()
        plt.show()
        self.fig_force.savefig('forces.png')

# %%
# Defining constants
R_earth = 6378  # km
h_leo = 500  # km
h_geo = 2000  # km
GM_earth = 398600  # km^3/s^2
v_circ = lambda r: np.sqrt(GM_earth / r)

# Defining inital and end conditions using polar coordinates (rho,theta) in earth centered inertial frame 
x_t0 = torch.tensor([R_earth + h_leo, 0]) 
x_tN = torch.tensor([R_earth + h_geo, 0])
v_t0 = torch.tensor([0, v_circ(R_earth + h_leo)])
v_tN = torch.tensor([0, v_circ(R_earth + h_geo)])

# Inital conditions in cartesian
x_t0_cart = torch.tensor([x_t0[0] * torch.cos(x_t0[1]), x_t0[0] * torch.sin(x_t0[1])])
x_tN_cart = torch.tensor([x_tN[0] * torch.cos(x_tN[1]), x_tN[0] * torch.sin(x_tN[1])])
v_t0_cart = torch.tensor([-v_t0[1] * torch.sin(x_t0[1]), v_t0[1] * torch.cos(x_t0[1])]) # radial part neglected 
v_tN_cart = torch.tensor([-v_tN[1] * torch.sin(x_tN[1]),  v_tN[1] * torch.cos(x_tN[1])])

def R(t, tN):
    R_cart = t * (x_tN_cart - x_t0_cart) + x_t0_cart
    return R_cart #R_cart.norm(dim=1), torch.atan2(R_cart[:,1], R_cart[:,0])

def R_dot(t, tN):
    R_dot_cart = (x_tN_cart - x_t0_cart) 
    return R_dot_cart #R_dot_cart.norm(), torch.atan2(R_dot_cart[1], R_dot_cart[0])

def V(t, tN):
    V_cart = t* (v_tN_cart - v_t0_cart) + v_t0_cart
    return V_cart #V_cart.norm(dim=1), torch.atan2(V_cart[:,1], V_cart[:,0])

def phi(t, tN):
    return 2*t**3 - 3*t**2 + t #1 - torch.cos(2*np.pi*t/tN)

def psi(t, tN):
    return t**2*(1 - t)**2

t_colloc = torch.linspace(0,1,1000).view(-1,1).requires_grad_(True)
t_total = torch.tensor(50, requires_grad=True, dtype=float)

ao_xyzgm = [[0, 0, GM_earth]]  # astronomic objects: x, y, gravitational mass

# Initialize model
pinn = PINN(1, 2, 50, 3, phi, psi, R, R_dot, V)
params = list(pinn.parameters())
params.append(t_total)
optimizer_adam = torch.optim.Adam(params, lr=1e-4)
optimizer_lbfgs = torch.optim.LBFGS(params, lr=.1, max_iter=10)

# Set random seed
seed = 2809
torch.manual_seed(seed)

obj = TrajectoryOptimizer(pinn, ao_xyzgm, t_colloc, t_total, x_t0, x_tN, optimizer_lbfgs, optimizer_adam, n_adam=200, n_lbfgs=0, w_physics=10, quiver_scale=20)

# %%
