#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformed.PINN import PINN as PINNTransformed
from transformed.TrajectoryOptimizer import TrajectoryOptimizer

from vanilla.TrajectoryOptimizer import TrajectoryOptimizer as TrajectoryOptimizerVanilla

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.unicode_minus": True,
    "font.size":11
}) 


# %%
