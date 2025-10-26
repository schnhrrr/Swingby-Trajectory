# %%

import torch
import numpy as np
from functools import partial

from ..config.config_3d import kinematic3d_config
from ..config.shared_parameters import x0_3d, xN_3d, v0_3d, vN_3d
from ..config.transform_functions import kinematic_fn
from ..runner import run_experiment, load_results
from ..plotter import TrajectoryPlotter

kinematic3d_config["optimizer"]["n_adam"] = 100  # 3000
kinematic3d_config["optimizer"]["n_lbfgs"] = 300

# Results from Vanilla PINN
v0_out = [3.0484906e-01, 1.1015986e00, 3.8672029e-03]
vN_out = [8.1449240e-01, 4.9920556e-01, 6.4733997e-03]

# Calculate input from calibrated model
a, b, c = load_results("kinematic_coeffs.pkl")
v_in_fn = lambda v_out: min(
    [r.real for r in np.roots([a, b, c - v_out]) if np.isreal(r)]
)

v0_in = torch.tensor(np.array([[v_in_fn(v) for v in v0_out]]))
vN_in = torch.tensor(np.array([[v_in_fn(v) for v in vN_out]]))

# v0_3d = torch.tensor([[3.05e-1,  1.10, 0.]])*2
# vN_3d = torch.tensor([[8.15e-1, 4.99e-1, 0.]])*2

kinematic3d_config["pinn"]["output_transform_fn"] = partial(
    kinematic_fn, x0=x0_3d, xN=xN_3d, v0=v0_in, vN=vN_in
)

results = []
for config in [kinematic3d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)


plotter = TrajectoryPlotter(results, dim=3, figsize=(6, 6))
plotter.plot_all()


# %%
