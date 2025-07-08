import torch
from src.pinn import PINN
from src.optimizer import TrajectoryOptimizer
from src.result import TrajectoryResult

def run_experiment(config):
    if "seed" in config:
        torch.manual_seed(config["seed"])
    model = PINN(**config['pinn'])  # Initialize the model

    # Register extra parameters if provided
    if extra_parameters := config['pinn'].get('extra_parameters', {}):
        for name, param in extra_parameters.items():
            model.register_parameter(str(name), param)
            config['optimizer'][name] = param  # Add to optimizer config
            
    opt = TrajectoryOptimizer(model, **config['optimizer'])  # Train the model
    res = TrajectoryResult(config['label'], opt)  # Extract the results
    return {"label": config['label'], 
            "result": res,
            **config.get("plotting",{}),
            "model": model}  # Return the results
