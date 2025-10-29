# %%
import numpy as np
from functools import partial

from swingby_trajectory.config.config_3d import orbit_transfer_config
from swingby_trajectory.runner import run_experiment

results = []
for config in [orbit_transfer_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from swingby_trajectory.plotter import TrajectoryPlotter

plotter = TrajectoryPlotter(results, dim=2, figsize=(6, 6), fig_prefix="orbit_transfer")
plotter.plot_all()

# %%
