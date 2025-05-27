#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config, vanilla3d_config
from src.runner import run_experiment

position3d_config['optimizer']['n_adam'] = 1000
position3d_config['optimizer']['n_lbfgs'] = 500
vanilla3d_config['optimizer']['n_adam'] = 2_000
vanilla3d_config['optimizer']['n_lbfgs'] = 1_000

results = []
for config in [position3d_config, vanilla3d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7))
plotter.plot_all()
