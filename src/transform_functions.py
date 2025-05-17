#%%
import torch

def R(t, x0, xN):
    return t*(xN - x0) + x0

def R_dot(t, x0, xN):
    return (xN - x0)

def V(t, v0, vN):
    return t*(vN - v0) + v0

def position_2d(t, x, x0, xN):
    """
    Transform function for the 2D position transformed PINN.
    """
    #x0, xN = torch.tensor([[-1., -1.]]), torch.tensor([[1., 1.]])
    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x      # phi = 0

def position_3d(t, x, x0, xN):
    """
    Transform function for the 3D position transformed PINN.
    """
    #x0, xN = torch.tensor([[-1., -1., 0.]]), torch.tensor([[1., 1., 0.]])
    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x      # phi = 0

def kinematic_2d(t, x, x0, xN, v0, vN):
    #x0, xN = torch.tensor([[-1., -1.]]), torch.tensor([[1., 1.]])
    #v0, vN = torch.tensor([1., 1.]), torch.tensor([1., 1.])
    psi = t**2 * (1 - t)**2
    phi = 2*t**3 - 3*t**2 + t
    return R(t, x0, xN) + psi * x + phi * (V(t, v0, vN) - R_dot(t, x0, xN))

def kinematic_3d(t, x, x0, xN, v0, vN):
    """
    Transform function for the 3D kinematic transformed PINN.
    """
    #x0, xN = torch.tensor([[-1., -1., 0.]]), torch.tensor([[1., 1., 0.]])
    #v0, vN = torch.tensor([1., 1., 0]), torch.tensor([1., 1., 0])
    psi = t**2 * (1 - t)**2
    phi = 2*t**3 - 3*t**2 + t
    return R(t, x0, xN) + psi * x + phi * (V(t, v0, vN) - R_dot(t, x0, xN))