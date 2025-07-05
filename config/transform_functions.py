import torch

def R(t, x0, xN):
    return t*(xN - x0) + x0

def R_dot(t, x0, xN):
    return (xN - x0)

def V(t, v0, vN):
    return t*(vN - v0) + v0

def position_fn(t, x, x0, xN):
    """
    Transform function for the 3D position transformed PINN.
    """
    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x      # phi = 0

def kinematic_fn(t, x, x0, xN, v0, vN):
    """
    Transform function for the 3D kinematic transformed PINN.
    """
    psi = t**2 * (1 - t)**2
    phi = 2*t**3 - 3*t**2 + t
    return R(t, x0, xN) + psi * x + phi * (V(t, v0, vN) - R_dot(t, x0, xN))