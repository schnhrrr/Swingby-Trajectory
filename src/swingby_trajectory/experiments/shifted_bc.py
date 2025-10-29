# %%

from functools import partial


from swingby_trajectory.config.config_3d import position3d_config
from swingby_trajectory.runner import run_experiment, export_results
from swingby_trajectory.config.transform_functions import position_fn
import torch
import numpy as np

x0_A = lambda A: torch.tensor([[-1.0, -1.0, A]])
xN_A = lambda A: torch.tensor([[1.0, 1.0, -A]])

position3d_config["optimizer"]["n_adam"] = 1_000
position3d_config["optimizer"]["n_lbfgs"] = 10_000
del position3d_config["plotting"]["color"]

results = []
for A in np.linspace(0.0, 1.0, 11):
    position3d_config["label"] = f"A={A:.1f}"
    position3d_config["pinn"]["output_transform_fn"] = partial(
        position_fn, x0=x0_A(A), xN=xN_A(A)
    )
    position3d_config["optimizer"]["r0"] = x0_A(A)
    position3d_config["optimizer"]["rN"] = xN_A(A)
    position3d_config["extra_parameters"]["t_total"] = torch.nn.Parameter(
        torch.tensor(1.0, requires_grad=True)
    )
    result = run_experiment(position3d_config)
    print(f"Training of {position3d_config['label']} completed.")
    results.append(result)

export_results(results, "shifted_bc_3d_results.pkl")

from swingby_trajectory.plotter import TrajectoryPlotter

plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7), fig_prefix="shifted_bc_3d")
plotter.plot_all(plot_quiver=False)
