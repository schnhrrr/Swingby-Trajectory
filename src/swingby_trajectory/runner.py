import pickle
import torch
from .pinn import PINN
from .optimizer import TrajectoryOptimizer
from .result import TrajectoryResult


def run_experiment(config):
    if "seed" in config:
        torch.manual_seed(config["seed"])
    model = PINN(**config["pinn"])  # Initialize the model

    # Register extra parameters if provided
    if extra_parameters := config.get("extra_parameters", {}):
        for name, param in extra_parameters.items():
            model.register_parameter(str(name), param)
            config["optimizer"][name] = param  # Add to optimizer config
            print("Trainable Parameter registered:\n", name, param, "\n")

    opt = TrajectoryOptimizer(model, **config["optimizer"])  # Train the model
    res = TrajectoryResult(config["label"], opt)  # Extract the results
    return {
        "label": config["label"],
        "result": res,
        **config.get("plotting", {}),
        "model": model,
    }  # Return the results


def export_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results
