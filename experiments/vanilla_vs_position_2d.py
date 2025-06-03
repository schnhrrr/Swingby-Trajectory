#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_2d import position2d_config, vanilla2d_config
from src.runner import run_experiment

vanilla2d_config['optimizer']['w_bc'] = 2.65

results = []
for config in [position2d_config, vanilla2d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=2, figsize=(6, 6))
plotter.plot_all()

# %%
