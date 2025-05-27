#%%
import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config
from src.runner import run_experiment
from config.transform_functions import position_3d
import torch


x0_A = lambda A: torch.tensor([[-1., -1., A]])
xN_A = lambda A: torch.tensor([[1., 1., -A]])

position3d_config['optimizer']['n_adam'] = 1000
position3d_config['optimizer']['n_lbfgs'] = 500
del position3d_config['plotting']['color']

results = []
for A in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    position3d_config['label'] = f"A={A}"
    position3d_config['pinn']['output_transform_fn'] = partial(position_3d, x0=x0_A(A), xN=xN_A(A))
    position3d_config['optimizer']['r0'] = x0_A(A)
    position3d_config['optimizer']['rN'] = xN_A(A)

    result = run_experiment(position3d_config)
    print(f"Training of {position3d_config['label']} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7))
plotter.plot_all()