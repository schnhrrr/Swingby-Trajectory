from functools import partial
import torch
from .transform_functions import position_fn, kinematic_fn
from .shared_parameters import (
    x0_3d,
    xN_3d,
    v0_3d,
    vN_3d,
    ao_3d,
    t_colloc,
    t_total,
)

position3d_config = {
    "label": "Position-transformed",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_fn, x0=x0_3d, xN=xN_3d),
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "solid",
        "color": "#1f77b4",
        "quiver_scale": 20,
    },
}

vanilla3d_config = {
    "label": "Vanilla",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2_000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 3.5,
    },
    "plotting": {
        "linestyle": "dashed",
        "color": "purple",
    },
}

kinematic3d_config = {
    "label": "Kinematic-transformed",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_fn, x0=x0_3d, xN=xN_3d, v0=v0_3d, vN=vN_3d
        ),
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 0,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#19907e",
    },
}

# Defining constants
import numpy as np

GM_earth = 398600e9  # km^3/s^2
R_earth = 6378e3  # km
h_leo = 500e3  # km
h_geo = 2000e3  # km
v_circ = lambda r: np.sqrt(GM_earth / r)

# Defining inital and end conditions using polar coordinates (rho,theta) in earth centered inertial frame
x0_c = torch.tensor([R_earth + h_leo, 0])
xN_c = torch.tensor([R_earth + h_geo, 0])
v0_c = torch.tensor([0, v_circ(R_earth + h_leo)])
vN_c = torch.tensor([0, v_circ(R_earth + h_geo)])

t_total_c = torch.tensor(3600 * 5).float()  # 5 hours

orbit_transfer_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_fn, x0=x0_c, xN=xN_c, v0=v0_c, vN=vN_c
        ),
    },
    "optimizer": {
        "ao_rgm": [[0, 0, GM_earth]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True),
        "t_total": t_total_c,
        "r0": x0_c,
        "rN": xN_c,
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
