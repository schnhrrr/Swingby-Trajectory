#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config, vanilla3d_config
from src.runner import run_experiment

if static_experiment := False:
    del vanilla3d_config['extra_parameters']
    del position3d_config['extra_parameters']
    position3d_config['optimizer']['n_adam'] = 1000
    position3d_config['optimizer']['n_lbfgs'] = 500
    vanilla3d_config['optimizer']['n_adam'] = 2_000
    vanilla3d_config['optimizer']['n_lbfgs'] = 500
    vanilla3d_config['optimizer']['w_bc'] = 68.36

vanilla3d_config['optimizer']['w_bc'] = 2.997
vanilla3d_config['optimizer']['n_adam'] = 2_000
vanilla3d_config['optimizer']['n_lbfgs'] = 10000

position3d_config['optimizer']['n_adam'] = 1000
position3d_config['optimizer']['n_lbfgs'] = 10000

results = []
for config in [position3d_config, vanilla3d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)
    print(f'E_kin,0 = {sum(result['result'].v[0]**2)/2}')

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7), fig_prefix="position_vanilla_3d")
plotter.plot_all()
