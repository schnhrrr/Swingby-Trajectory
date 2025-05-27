from functools import partial
import torch
from config.transform_functions import position_3d, kinematic_3d
from config.shared_parameters import x0_3d, xN_3d, ao_3d, t_colloc, t_total

position3d_config = {
    "label": "Position-transformed",
    "seed": 2809,
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_3d, x0=x0_3d, xN=xN_3d),
    },
    "optimizer":{
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 0,
        "n_lbfgs": 200,
        "w_physics": 1.,
        "w_bc": 0,
    },
    "plotting":{
        "linestyle": "solid",
        "color": "#1f77b4",
        "quiver_scale": 20,
    }
}    

vanilla3d_config = {
    "label": "Vanilla",
    "seed": 2809,
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer":{
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2_000,
        "n_lbfgs": 700,
        "w_physics": 1.,
        "w_bc": 3.5,
    },
    "plotting":{
        "linestyle": "dashed",
        "color": "purple",
    }
}