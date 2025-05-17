from src.pinn import PINN
from src.optimizer import TrajectoryOptimizer
from src.result import TrajectoryResult

def run_experiment(config):
    model = PINN(**config['pinn'])  # Build the PINN
    opt = TrajectoryOptimizer(model, **config['optimizer'])  # Train the model
    res = TrajectoryResult(config['label'], opt)  # Extract the results
    return {"label": config['label'], 
            "result": res,
            **config.get("plotting",{})}  # Return the results
