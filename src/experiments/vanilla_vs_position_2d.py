#%%
from experiments.config.config_2d import position2d_config, vanilla2d_config
from runner import run_experiment

results = []
for config in [position2d_config, vanilla2d_config]:
    result = run_experiment(config)
    print(f"Training of {config['label']} completed.")
    results.append(result)

from plotter import TrajectoryPlotter


# %%
