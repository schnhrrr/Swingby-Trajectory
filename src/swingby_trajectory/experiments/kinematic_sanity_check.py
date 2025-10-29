# %%
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch

from swingby_trajectory.runner import run_experiment
from swingby_trajectory.plotter import TrajectoryPlotter
from swingby_trajectory.config.config_3d import position3d_config, kinematic3d_config
from swingby_trajectory.config.shared_parameters import x0_3d, xN_3d
from swingby_trajectory.config.transform_functions import kinematic_fn

position3d_config["optimizer"]["n_adam"] = 2_000
kinematic3d_config["optimizer"]["n_adam"] = 2_000
# LBFGS stops after convergence

pos_result = run_experiment(position3d_config)
v0_pos, vN_pos = pos_result["result"].v[0, :], pos_result["result"].v[-1, :]
print(f"Training of {position3d_config['label']} completed.\n")

kinematic3d_config["pinn"]["output_transform_fn"] = partial(
    kinematic_fn,
    x0=x0_3d,
    xN=xN_3d,
    v0=torch.from_numpy(v0_pos).view(1, -1),
    vN=torch.from_numpy(vN_pos).view(1, -1),
)
kin_result = run_experiment(kinematic3d_config)
v0_kin, vN_kin = kin_result["result"].v[0, :], kin_result["result"].v[-1, :]
print(f"Training of {kinematic3d_config['label']} completed.\n")

print("Position PINN boundary velocities:")
print(f"  v0: {v0_pos}")
print(f"  vN: {vN_pos}\n")
print("Kinematic PINN boundary velocities:")
print(f"  v0: {v0_kin}")
print(f"  vN: {vN_kin}\n")
print("Difference in boundary velocities:")
print(f"  delta v0: {v0_kin - v0_pos}")
print(f"  delta vN: {vN_kin - vN_pos}")

pos_result["color"] = "#1f77b4"  # Blue
kin_result["color"] = "#ff7f0e"  # Orange
plotter = TrajectoryPlotter(
    experiments=[pos_result, kin_result],
    dim=3,
    figsize=(7, 7),
    fig_prefix="kinematic_sanity_check_3d",
)
plotter.plot_all()

# %%
