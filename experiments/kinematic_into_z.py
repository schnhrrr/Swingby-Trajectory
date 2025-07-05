#%%
import os
import sys
import torch
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_3d import position3d_config, vanilla3d_config, kinematic3d_config
from src.runner import run_experiment

kinematic3d_config['optimizer']['n_adam'] = 2000
kinematic3d_config['optimizer']['n_lbfgs'] = 150
v0_3d = torch.tensor([[ 0.2479,  0.9203,  1.2174]])
vN_3d = torch.tensor([[ 0.5784,  0.3899, -1.2156]])

kinematic3d_config['optimizer']['output_transform_fn'] = partial(
    kinematic3d_config['pinn']['output_transform_fn'],
    x0=kinematic3d_config['optimizer']['r0'],
    xN=kinematic3d_config['optimizer']['rN'],
    v0=v0_3d,
    vN=vN_3d)

results = []
for config in [kinematic3d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7))
plotter.plot_all()

# %%
