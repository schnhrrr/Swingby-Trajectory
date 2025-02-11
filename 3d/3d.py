#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
eps = 1e-8

# Architecture of NN
n_input     = 1 # normalized time
n_output    = 3 # x,y,z
n_layers    = 3
n_neurons   = 50
lr          = 1.e-4

# Define NN class
class NN(nn.Module):
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
        g = torch.cat([2*t-1, 2*t-1, 2*t-1], dim=1)
        phi = torch.cat([t*(1-t),t*(1-t),t*(1-t)], dim=1)
        xyz_transformed = g + (phi + eps)* xyz
        return xyz_transformed



def plotTrajectory(r):
    # Extract x, y, z coordinates from tensor and convert to NumPy arrays
    x, y, z = r[:, 0].detach().numpy(), r[:, 1].detach().numpy(), r[:, 2].detach().numpy()

    # Create a figure with 2x2 subplots
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Plot x-y trajectory in the first subplot (top-left)
    ax[0, 0].plot(x, y, label='Trajectory')

    # Create a 3D subplot in the bottom-right position
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')

    # Loop through gravitational mass data (ao_xygm) and plot their positions
    i = 1
    for xtemp, ytemp, ztemp, gmtemp in ao_xygm:
        ax[0, 0].scatter(xtemp, ytemp, s=gmtemp * 200, c='k', marker='o')
        ax[0, 0].text(xtemp, ytemp + 0.1, f'GM$_{i}$')
        ax[1, 0].scatter(xtemp, ztemp, s=gmtemp * 200, c='k', marker='o')
        ax[1, 0].text(xtemp, ztemp + 0.1, f'GM$_{i}$')
        ax[0, 1].scatter(ytemp, ztemp, s=gmtemp * 200, c='k', marker='o')
        ax[0, 1].text(ytemp, ztemp + 0.1, f'GM$_{i}$')
        ax3d.scatter(xtemp, ytemp, ztemp, s=gmtemp * 100, c='k', marker='o')
        ax3d.text(xtemp, ytemp, ztemp + 0.1, f'GM$_{i}$')
        i += 1

    # x-y plot
    ax[0, 0].set_xlabel(r'$x$')
    ax[0, 0].set_ylabel(r'$y$')
    ax[0, 0].scatter(-1, -1, color='r', marker='x', label='BC Start')
    ax[0, 0].scatter(1, 1, color='r', marker='x', label='BC End')
    ax[0, 0].legend()

    # x-z plot
    ax[1, 0].plot(x, z, label='xz')
    ax[1, 0].scatter(-1, -1, color='r', marker='x', label='BC Start')
    ax[1, 0].scatter(1, 1, color='r', marker='x', label='BC End')
    ax[1, 0].set_xlabel(r'$x$')
    ax[1, 0].set_ylabel(r'$z$')

    # Plot y-z trajectory (top-right subplot)
    ax[0, 1].plot(y, z, label='yz')
    ax[0, 1].scatter(-1, -1, color='r', marker='x', label='BC Start')
    ax[0, 1].scatter(1, 1, color='r', marker='x', label='BC End')
    ax[0, 1].set_xlabel(r'$y$')
    ax[0, 1].set_ylabel(r'$z$')

    # Remove the unused subplot (bottom-right)
    fig.delaxes(ax[1, 1])

    # 3D plot
    bc = np.ones(3)
    ax3d.scatter(*-bc, marker='x', color='r')
    ax3d.scatter(*bc, marker='x', color='r')
    ax3d.plot3D(x, y, z)
    ax3d.set_xlabel(r'$x$')
    ax3d.set_ylabel(r'$y$')
    ax3d.set_zlabel(r'$z$')
    ax3d.view_init(elev=25, azim=135)
    fig.show()
   

def plotGravityAndThrust(r, Thrust):
    x, y, z = r[:,0].detach().numpy(), r[:,1].detach().numpy(), r[:,2].detach().numpy()
    Thrust = Thrust.detach().numpy()
    Gx = 0
    Gy = 0
    Gz = 0
    for xtemp, ytemp, ztemp, gmtemp in ao_xygm:
        Gx += gmtemp * m0 * (x - xtemp) / ((x - xtemp)**2 + (y - ytemp)**2 + (z - ztemp)**2 + eps)**1.5
        Gy += gmtemp * m0 * (y - ytemp) / ((x - xtemp)**2 + (y - ytemp)**2 + (z - ztemp)**2 + eps)**1.5
        Gz += gmtemp * m0 * (z - ztemp) / ((x - xtemp)**2 + (y - ytemp)**2 + (z - ztemp)**2 + eps)**1.5
    
    t = np.linspace(0, 1, len(x))
    G = np.sqrt((Gx**2 + Gy**2 + Gz**2)) 

    plt.figure()
    plt.plot(t, G, label="Total gravity" , color='k')
    plt.plot(t, Thrust, label=r"Thrust magnitude")
    plt.fill_between(t,np.zeros_like(Thrust[:,0]),Thrust[:,0])
    plt.plot(t, G.reshape(-1,1) + Thrust, label="Required force magnitude", linestyle="--")
    plt.xlabel('Normalized time')
    plt.ylabel('Exerted force')
    plt.xlim(0,1)
    plt.plot([0,1],[0,0],linestyle='--',color='k')
    plt.legend()
    plt.show()

# Thrust vector
def ode(t, r):
    x, y, z    = r[:,0], r[:,1], r[:,2]
    dxdt    = torch.autograd.grad(x, t, create_graph=True, grad_outputs=torch.ones_like(x))[0] / T
    dxdt2   = torch.autograd.grad(dxdt, t, create_graph=True, grad_outputs=torch.ones_like(dxdt))[0] / T
    dydt    = torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones_like(y))[0] / T
    dydt2   = torch.autograd.grad(dydt, t, create_graph=True, grad_outputs=torch.ones_like(dydt))[0] / T
    dzdt    = torch.autograd.grad(z, t, create_graph=True, grad_outputs=torch.ones_like(z))[0] / T
    dzdt2   = torch.autograd.grad(dzdt, t, create_graph=True, grad_outputs=torch.ones_like(dzdt))[0] / T

    ode_x = (m0 * dxdt2).view(1,-1)
    ode_y = (m0 * dydt2).view(1,-1)
    ode_z = (m0 * dzdt2).view(1,-1)

    for xtemp, ytemp, ztemp, gmtemp in ao_xygm:
        x_diff = x - xtemp
        y_diff = y - ytemp
        z_diff = z - ztemp
        denominator = (x_diff**2 + y_diff**2 + z_diff**2 + eps)**1.5
        ode_x += gmtemp * m0 * x_diff / denominator
        ode_y += gmtemp * m0 * y_diff / denominator
        ode_z += gmtemp * m0 * z_diff / denominator
    
    return ode_x.view(-1,1), ode_y.view(-1,1), ode_z.view(-1,1)

def v(t, r):
    x, y, z    = r[:,0], r[:,1], r[:,2]
    dxdt    = torch.autograd.grad(x, t, create_graph=True, grad_outputs=torch.ones_like(x))[0] / T
    dydt    = torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones_like(y))[0] / T
    dzdt    = torch.autograd.grad(z, t, create_graph=True, grad_outputs=torch.ones_like(z))[0] / T

    return torch.sqrt(dxdt**2 + dydt**2 + dzdt**2)

# Set random seed
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

# Physical Parameters
t_colloc    = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)
r0          = torch.tensor([[-1.,-1.,-1.]]) # start point
r1          = torch.tensor([[1.,1.,1.]]) # end point
m0          = 1.
T           = torch.tensor(1.0, requires_grad=True) # end time

ao_xygm     =  [[-0.5,  -1., -1., 0.7],  # astronomic objects: x, y, z gravitational mass
                [-0.2,  0.4, 0., 1.0],
                [ 0.8,  0.3, 0.5, 0.6]]


# Initialize model
model = NN(n_input, n_output, n_neurons, n_layers)
parameters = list(model.parameters())
parameters.append(T)
optimizer_adam = torch.optim.Adam(parameters, lr=lr)
optimizer_lbfgs = torch.optim.LBFGS(parameters, lr=.1, max_iter=10)

# Initialize
loss_BC_history = []
loss_physics_history = []
loss_history = []

# Hyperparameters
w_physics = 1

loops_adam = 0
loops_lbfgs = 150
# Begin training
for i in range(loops_adam + loops_lbfgs):
    if i % 10 == 0: 
        print('Epoche: ',i)

    def closure():
        optimizer.zero_grad()

        # Compute loss from physics
        r_model = model(t_colloc)
        Fx, Fy, Fz = ode(t_colloc, r_model)
        Thrust = Fx**2 + Fy**2 + Fz**2 + eps
        loss_physics = torch.mean(Thrust)
        loss = w_physics * loss_physics


        if (i+1) % 50 == 0:
            plotTrajectory(r_model)
            plotGravityAndThrust(r_model, torch.sqrt(Thrust))
        loss_physics_history.append(loss_physics.detach())
        loss_history.append(loss.detach())
            
        loss.backward()
        return loss

    # 3000 iterations ADAM, 100 iterations LBFGS
    if i<loops_adam:
        optimizer = optimizer_adam
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()

    else:
        optimizer = optimizer_lbfgs
        optimizer.step(closure)   


    # Visualize training progress
    if (i+1) % (loops_adam + loops_lbfgs) == 0:
        r_model = model(t_colloc)
        Fx, Fy, Fz = ode(t_colloc, r_model)
        Thrust = torch.sqrt(Fx**2 + Fy**2 + Fz**2)
        plotTrajectory(r_model)
        plotGravityAndThrust(r_model, Thrust)


# Plotting loss history
plt.figure()
plt.legend()
plt.plot(loss_physics_history, label="Physics Loss")
plt.yscale('log')
plt.legend()
plt.xlim(left=0)
plt.xlabel('Training epoch')
plt.ylabel('Loss')

print(model.activation, f'lr: {lr}, w_physics: {w_physics} loops_adam: {loops_adam}, loops_BFGS: {loops_lbfgs}, arch: {n_layers, n_neurons}')

