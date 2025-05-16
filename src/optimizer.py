import torch
#%%
class TrajectoryOptimizer:
    """" Class to optimize the trajectory of a spacecraft using a physics-informed neural network (PINN) denoted in the model attribute.
    The optimizer uses the PINN to compute the gravity and thrust forces acting on the spacecraft, and then 
    optimizes the trajectory in a way that no thrust is needed by the spacecraft. So the thrust is equal to the gravity force.
    The optimizer minimizes the loss function, which is a combination of the physics loss and the boundary condition loss.
    
    Args:
        model (PINN): The PINN model used to compute the trajectory.
        ao_rgm (list): List of lists containing the position and mass of the astronomical objects.
        t_colloc (torch.Tensor): Collocation points for the trajectory.
        r0 (torch.Tensor): Initial position of the spacecraft.
        rN (torch.Tensor): Final position of the spacecraft.
        opt_adam (torch.optim.Optimizer): Adam optimizer for the initial optimization.
        opt_lbfgs (torch.optim.Optimizer): LBFGS optimizer for the final optimization.
        n_adam (int, optional): Number of iterations for the Adam optimizer. Defaults to 0.
        n_lbfgs (int, optional): Number of iterations for the LBFGS optimizer. Defaults to 100.
        w_physics (float, optional): Weight for the physics loss. Defaults to 1.
        w_bc (float, optional): Weight for the boundary condition loss. Defaults to 0.

    """
    # TODO: add typehints
    def __init__(self, model, ao_rgm, t_colloc, t_total, r0, rN, opt_adam, opt_lbfgs, n_adam=0, n_lbfgs=100, w_physics=1, w_bc=0):
        self.model = model  # Pinn
        self.eps = 1e-8
        self.lbfgs = opt_lbfgs
        self.adam = opt_adam
        self.n_lbfgs = n_lbfgs
        self.n_adam = n_adam

        self.w_physics = w_physics
        self.w_bc = w_bc

        self.loss_history = []
        if w_bc > 0:
            self.loss_physics_history = []
            self.loss_bc_history = []

        self.t = t_colloc  # Collocation points
        self.t_total = t_total # Time to go from r0 to rN
        self.r0 = r0  # r(t=0)
        self.rN = rN  # r(t=1)
        self.dims = self.r0.shape[-1]  # Number of dimensions
        self.ao_rgm = torch.tensor(ao_rgm)  # Astronomic objects (r, Gm)
        
        self._train_model() 

    def _compute_gravitational_force(self):

        self.G = torch.zeros_like(self.r)
        for ao in self.ao_rgm:
            r_diff = self.r - ao[:-1]
            denominator = (torch.linalg.norm(r_diff, dim=1) + self.eps)**3
            for i in range(self.dims):
                self.G[:,i] -= ao[-1] * r_diff[:,i] / denominator

    def _compute_thrust(self):
        
        self.v = torch.stack([torch.autograd.grad(self.r[:, i], self.t, grad_outputs=torch.ones_like(self.r[:, i]), create_graph=True)[0].squeeze(-1)
                 for i in range(self.dims)], dim=1) / self.t_total
        
        self.a = torch.stack([torch.autograd.grad(self.v[:, i], self.t, grad_outputs=torch.ones_like(self.v[:, i]), create_graph=True)[0].squeeze(-1)
                 for i in range(self.dims) ], dim=1) / self.t_total
            
        self._compute_gravitational_force()
        self.T = self.a - self.G 
        
    def _train_model(self):

        for i in range(self.n_adam + self.n_lbfgs):
            if i % 20 == 0 and i < self.n_adam:
                print(f'ADAM: {i}')
            elif i % 10 == 0 and i > self.n_adam:
                print(f'LBFGS: {i}')
            if i < self.n_adam:
                self.optimizer = self.adam
                self.optimizer.zero_grad()
                self._closure()
                self.optimizer.step()
            else:
                self.optimizer = self.lbfgs
                self.optimizer.step(self._closure) 

        print("Trajectory optimization finished.")

    def _closure(self):

        self.optimizer.zero_grad()
        self.r = self.model(self.t) 
        self._compute_thrust()
        self.loss_physics = torch.mean(self.T.norm(dim=1)) * self.w_physics
        
        if self.w_bc > 0:
            self.loss_bc = (torch.mean((self.r[0] - self.r0)**2) + torch.mean((self.r[-1] - self.rN)**2)) / 2 * self.w_bc
            self.loss = self.loss_bc + self.loss_physics

            self.loss_physics_history.append(self.loss_physics.item())
            self.loss_bc_history.append(self.loss_bc.item())
            self.loss_history.append(self.loss.item())
            self.loss.backward()
            return self.loss
        else:
            self.loss_history.append(self.loss_physics.item())
            self.loss_physics.backward()
            return self.loss_physics
        