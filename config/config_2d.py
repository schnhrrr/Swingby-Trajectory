from functools import partial
import torch
from config.transform_functions import position_fn, kinematic_fn
from config.shared_parameters import x0_2d, xN_2d, ao_2d, t_colloc, t_total

position2d_config = {
    "label": "Position-transformed",
    "seed": 2809,
    "extra_parameters": {
            "t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_fn, x0=x0_2d, xN=xN_2d),
    },
    "optimizer":{
        "ao_rgm": ao_2d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_2d,
        "rN": xN_2d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 1000,
        "n_lbfgs": 500,
        "w_physics": 1.,
        "w_bc": 0,
    },
    "plotting":{
        "linestyle": "solid",
        "color": "#191970",  # MidnightBlue
        "quiver_scale": 20,
    }
}    

vanilla2d_config = {
    "label": "Vanilla",
    "seed": 2809,
    "extra_parameters": {
            "t_total": torch.nn.Parameter(t_total)},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer":{
        "ao_rgm": ao_2d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_2d,
        "rN": xN_2d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 3000,
        "n_lbfgs": 400,
        "w_physics": 1.,
        "w_bc": 3.5,
    },
    "plotting":{
        "linestyle": "dashed",
        "color": "#008080",  # Teal
    }
}