import numpy as np
from optimizer import TrajectoryOptimizer

class TrajectoryResult:
    """Class to extract and store the results of the trajectory optimization."""

    def __init__(self, label: str, traj_opt: TrajectoryOptimizer):
        self.label = label # Expriment label

        self.t = traj_opt.t.detach().numpy()
        self.t_total = traj_opt.t_total.item()

        self.ao = traj_opt.ao_xyzgm.detach().numpy()
        self.r0 = traj_opt.r0.detach().numpy()
        self.r1 = traj_opt.r1.detach().numpy()

        self.r = traj_opt.r.detach().numpy()
        self.v = traj_opt.v.detach().numpy()
        self.a = traj_opt.a.detach().numpy()

        self.G = traj_opt.G.detach().numpy()
        self.T = traj_opt.T.detach().numpy()
        
        # Magnitudes 
        self.a_mag = np.linalg.norm(self.a, axis=1)
        self.G_mag = np.linalg.norm(self.G.norm, axis=1)
        self.T_mag = np.linalg.norm(self.T.norm, axis=1)

        self.loss_history = [l.item() for l in traj_opt.loss_history]
        self.loss_physics_history = [l.item() for l in getattr(traj_opt, 'loss_physics_history', [])]
        self.loss_bc_history = [l.item() for l in getattr(traj_opt, 'loss_bc_history', [])]
