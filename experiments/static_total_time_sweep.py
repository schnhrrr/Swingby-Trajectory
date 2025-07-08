#%%
import os
import sys
import torch
import numpy as np
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_3d import position3d_config
from src.runner import run_experiment

position3d_config['optimizer']['n_adam'] = 1000
position3d_config['optimizer']['n_lbfgs'] = 500

# Sweeping over different STATIC (non-trainable) total times
t_total_vec = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

results = []
for t_total in t_total_vec:
    position3d_config['label'] = f"Total Time: {t_total:.2f} s"
    position3d_config['optimizer']['t_total'] = torch.tensor(t_total, requires_grad=False)
    result = run_experiment(position3d_config)
    print(f"Training of {t_total} completed.")
    results.append(result)

from src.plotter import TrajectoryPlotter
plotter = TrajectoryPlotter(results, dim=3, figsize=(7, 7))
plotter.plot_all()

# TODO: Add prefix as input to plotter: if not prefix: generate_figname()
# TODO: colors
# TODO: loss colors
# TODO: pass label as str(float) and pass figname prefix seperately
# TODO: RFM, G, T
import matplotlib.pyplot as plt
def get_color_palette(num_colors):
    return plt.cm.viridis(np.linspace(0,1,num_colors))
cols = get_color_palette(results.__len__)
for res in results:
    res['label'] = str(res['result'].t_total).replace('.',',') 