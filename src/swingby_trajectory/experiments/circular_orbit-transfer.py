# %%
import torch
import numpy as np
from functools import partial

from swingby_trajectory.config.config_3d import kinematic3d_config
from swingby_trajectory.config.shared_parameters import (
    x0_3d,
    xN_3d,
    v0_3d,
    vN_3d,
    t_total,
    t_colloc,
)
from swingby_trajectory.config.transform_functions import kinematic_fn
from swingby_trajectory.runner import run_experiment, load_results

# Defining constants
R_earth = 6378e3  # km
h_leo = 500e3  # km
h_geo = 2000e3  # km
GM_earth = 398600e9  # km^3/s^2
v_circ = lambda r: np.sqrt(GM_earth / r)

# Defining inital and end conditions using polar coordinates (rho,theta) in earth centered inertial frame
x_t0 = torch.tensor([R_earth + h_leo, 0])
x_tN = torch.tensor([R_earth + h_geo, 0])
v_t0 = torch.tensor([0, v_circ(R_earth + h_leo)])
v_tN = torch.tensor([0, v_circ(R_earth + h_geo)])

# Inital conditions in cartesian
x0 = torch.tensor([x_t0[0] * torch.cos(x_t0[1]), x_t0[0] * torch.sin(x_t0[1])])
xN = torch.tensor([x_tN[0] * torch.cos(x_tN[1]), x_tN[0] * torch.sin(x_tN[1])])
v0 = torch.tensor(
    [-v_t0[1] * torch.sin(x_t0[1]), v_t0[1] * torch.cos(x_t0[1])]
)  # radial part neglected
vN = torch.tensor([-v_tN[1] * torch.sin(x_tN[1]), v_tN[1] * torch.cos(x_tN[1])])

t_total = torch.tensor([[100000.0]])

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
        "output_transform_fn": partial(kinematic_fn, x0=x0, xN=xN, v0=v0, vN=vN),
    },
    "optimizer": {
        "ao_rgm": [[0, 0, GM_earth]],
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0,
        "rN": xN,
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

results = []
for config in [orbit_transfer_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from swingby_trajectory.plotter import TrajectoryPlotter

plotter = TrajectoryPlotter(results, dim=2, figsize=(6, 6), fig_prefix="orbit_transfer")
plotter.plot_all()
