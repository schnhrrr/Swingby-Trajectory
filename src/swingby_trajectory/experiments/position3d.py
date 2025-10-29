# %%

from swingby_trajectory.config.config_3d import position3d_config
from swingby_trajectory.runner import run_experiment

import random

results = []
seeds = random.sample(range(1, 1000), 5)
for seed in seeds:
    print(f"Running experiment with seed {seed}")
    config = position3d_config.copy()
    config["label"] = f"Seed {seed}"
    config["seed"] = seed
    config["optimizer"]["n_adam"] = 1000
    config["optimizer"]["n_lbfgs"] = 100
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from swingby_trajectory.plotter import TrajectoryPlotter

plotter = TrajectoryPlotter(results, dim=3, figsize=(6, 6))
plotter.plot_traj_3d(plot_quiver=False)
