#%%
import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config
from src.runner import run_experiment, export_results
from config.transform_functions import position_fn
import torch
import numpy as np

x0_A = lambda A: torch.tensor([[-1., -1., A]])
xN_A = lambda A: torch.tensor([[1., 1., -A]])

position3d_config['optimizer']['n_adam'] = 1_000
position3d_config['optimizer']['n_lbfgs'] = 10_000
del position3d_config['plotting']['color']

results = []
for A in np.linspace(0.0, 1.0, 11):
    position3d_config['label'] = f"A={A:.1f}"
    position3d_config['pinn']['output_transform_fn'] = partial(position_fn, x0=x0_A(A), xN=xN_A(A))
    position3d_config['optimizer']['r0'] = x0_A(A)
    position3d_config['optimizer']['rN'] = xN_A(A)

    result = run_experiment(position3d_config)
    print(f"Training of {position3d_config['label']} completed.")
    results.append(result)

export_results(results, "shifted_bc_3d_results.pkl")

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7), fig_prefix="shifted_bc_3d")
plotter.plot_all(plot_quiver=False)