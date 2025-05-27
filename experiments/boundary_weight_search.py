#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from config.config_2d import vanilla2d_config
from src.runner import run_experiment

vanilla2d_config['optimizer']['n_adam'] = 2000
vanilla2d_config['optimizer']['n_lbfgs'] = 500

idx = []
w_bc_list = []
min_bc_loss = []
min_total_loss = []

weights = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 1_000))

for i, w_bc in enumerate(weights):
    print(i)
    vanilla2d_config['optimizer']['w_bc'] = w_bc
    result = run_experiment(vanilla2d_config)
    idx.append(i)
    min_bc_loss.append(min(result['result'].loss_bc))
    min_total_loss.append(min(result['result'].loss))
    w_bc_list.append(w_bc)

data = np.column_stack([idx, w_bc_list, min_bc_loss, min_total_loss])
np.savetxt("weights.csv", data, delimiter=",", fmt="%.4f", header="Idx,w_bc,min-bc-loss,min-total-loss", comments='')

