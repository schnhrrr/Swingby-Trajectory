#%%
import os
import sys
import torch
import numpy as np
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config
from src.plotter import TrajectoryPlotter
from src.runner import run_experiment, export_results

position3d_config['optimizer']['n_adam'] = 1000
position3d_config['optimizer']['n_lbfgs'] = 10_000
del position3d_config['extra_parameters']

# Sweeping over different STATIC (non-trainable) total times
t_total_small = [0.33, 0.5, 0.66, 0.75, 0.9, 1.0]
t_total_medium = np.linspace(1.0, 1.1, 6).tolist()
t_total_large = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]

results = {}
for i, t_total_vec in enumerate([t_total_small, t_total_medium, t_total_large]):
    results_temp = []
    for t_total in t_total_vec:
        position3d_config['label'] = f"Total Time: {t_total:.3f} s"
        position3d_config['optimizer']['t_total'] = torch.tensor(t_total, requires_grad=False)
        result = run_experiment(position3d_config)
        print(f"Training of {t_total} completed.")
        del result['color']
        results_temp.append(result)

    plotter = TrajectoryPlotter(results_temp, fig_prefix=f'static_time_sweep_{i}', dim=3, figsize=(7, 7))
    plotter.plot_all(plot_quiver=False)
    results[i] = [results_temp, plotter]

export_results(results, "static_total_time_sweep_results.pkl")
