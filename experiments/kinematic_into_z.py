#%%
import os
import sys
import torch
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import kinematic3d_config
from config.shared_parameters import x0_3d, xN_3d, v0_3d, vN_3d
from config.transform_functions import kinematic_fn
from src.runner import run_experiment

kinematic3d_config['optimizer']['n_adam'] = 50 #3000
kinematic3d_config['optimizer']['opt_adam'] = partial(torch.optim.Adam, lr=1e-3)
kinematic3d_config['optimizer']['n_lbfgs'] = 1

#v0_3d = torch.tensor([[ 1., 1., 1.]])
#vN_3d = torch.tensor([[1., 1., -1.]])
v0_3d = torch.tensor([[ 0.4308,  1.0814,  1.1313]])
vN_3d = torch.tensor([[ 0.7383,  0.5656, -1.1591]])
#v0_3d = torch.tensor([[ 0.8,  1.4,  -1.]])
#vN_3d = torch.tensor([[ 1.2,  0.6, 1.]])

import math
sq2 = math.sqrt(2)*2
v0_3d = torch.tensor([[sq2, sq2,0.]])
vN_3d = torch.tensor([[sq2, sq2,0.]])

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
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7))
plotter.plot_all()
