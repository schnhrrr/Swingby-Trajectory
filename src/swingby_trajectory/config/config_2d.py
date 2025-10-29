from functools import partial
import torch
from .transform_functions import position_fn, kinematic_fn
from .shared_parameters import (
    x0_2d,
    xN_2d,
    ao_2d,
    t_colloc,
    t_total,
)

position2d_config = {
    "label": "Position-transformed",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_fn, x0=x0_2d, xN=xN_2d),
    },
    "optimizer": {
        "ao_rgm": ao_2d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_2d,
        "rN": xN_2d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 3000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "solid",
        "color": "#191970",  # MidnightBlue
        "quiver_scale": 20,
    },
}

vanilla2d_config = {
    "label": "Vanilla",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer": {
        "ao_rgm": ao_2d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_2d,
        "rN": xN_2d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 3000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 3.5,
    },
    "plotting": {
        "linestyle": "dashed",
        "color": "#008080",  # Teal
    },
}

# Defining constants
import numpy as np

GM_earth = 398600.0  # km^3/s^2
R_earth = 6378.0  # km
h_leo = 500.0  # km
h_heo = 2000.0  # km
omega = lambda r: np.sqrt(GM_earth / r**3)
phi_trainable = torch.nn.Parameter(
    torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)
)

R_leo = R_earth + h_leo
R_heo = R_earth + h_heo
omega_leo = omega(R_leo)
omega_heo = omega(R_heo)

# Inital conditions in cartesian
x0_ot = torch.tensor([R_leo, 0])
xN_ot = torch.tensor(
    [R_heo * torch.cos(phi_trainable), R_heo * torch.sin(phi_trainable)]
)
v0_ot = torch.tensor([0, omega_leo * R_leo])  # No radial velocity
vN_ot = (
    torch.tensor([-R_heo * torch.sin(phi_trainable), R_heo * torch.cos(phi_trainable)])
    * omega_heo
)


t_total_ot = torch.tensor(3600 * 5).float()  # 5 hours

orbit_transfer_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "extra_parameters": {
        "t_total": torch.nn.Parameter(t_total),
        "phi_trainable": phi_trainable,
    },
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_fn, x0=x0_ot, xN=xN_ot, v0=v0_ot, vN=vN_ot
        ),
    },
    "optimizer": {
        "ao_rgm": [[0, 0, GM_earth]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True),
        "t_total": t_total_ot,
        "r0": x0_ot,
        "rN": xN_ot,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#19907e",
    },
}
