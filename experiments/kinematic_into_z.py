#%%
import os
import sys
import torch
import numpy as np
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import kinematic3d_config
from config.shared_parameters import x0_3d, xN_3d, v0_3d, vN_3d
from config.transform_functions import kinematic_fn
from src.runner import run_experiment, load_results

kinematic3d_config['optimizer']['n_adam'] = 1000 
kinematic3d_config['optimizer']['n_lbfgs'] = 10_000

v0_3d = torch.tensor([[1.4155664, 2.0415726, 0.]])
vN_3d = torch.tensor([[1.7794383, 1.5514505, 0.]])

kinematic3d_config['pinn']['output_transform_fn'] = partial(
    kinematic_fn,
    x0=x0_3d,
    xN=xN_3d,
    v0=v0_3d,
    vN=vN_3d)

results = []
for config in [kinematic3d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(6, 6))
plotter.plot_all()


# %%
