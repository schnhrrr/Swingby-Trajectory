#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_2d import position2d_config, vanilla2d_config
from src.runner import run_experiment

if static_experiment := False:  # Set True for static time experiment
    del position2d_config['extra_parameters']
    del vanilla2d_config['extra_parameters']
    vanilla2d_config['optimizer']['w_bc'] = 2.65
    position2d_config['optimizer']['n_adam'] = 200
    position2d_config['optimizer']['n_lbfgs'] = 200

vanilla2d_config['optimizer']['w_bc'] = 1.67

results = []
for config in [position2d_config, vanilla2d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)
    print(f'E_kin,0 = {sum(result['result'].v[0]**2)/2}')

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=2, figsize=(7, 7), fig_prefix="position_vanilla_2d")
plotter.plot_all()

# %%
