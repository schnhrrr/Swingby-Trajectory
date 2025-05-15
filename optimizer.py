import torch

class TrajectoryOptimizer:

    def __init__(self, model, ao_xyzgm, t_colloc, r0, r1, opt_adam, opt_lbfgs, n_adam=0, n_lbfgs=100, w_physics=1, w_bc=0):
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
        self.t_total = list(model.parameters())[-1]  # Time to go from r0 to r1
        self.r0 = r0  # r(t=0)
        self.r1 = r1  # r(t=1)
        self.dims = self.r0.shape[-1]  # Number of dimensions
        self.ao_xyzgm = torch.tensor(ao_xyzgm)  # Astronomic objects (r, Gm)

        self.r = self.model(self.t)  # Initial guess for the trajectory
        self.G = torch.zeros_like(self.r)  # Gravitational force
        
        self._train_model()

    def _compute_gravitational_force(self):

        self.G.zero_()
        for ao in self.ao_xyzgm:
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
                self.closure()
                self.optimizer.step()
            else:
                self.optimizer = self.lbfgs
                self.optimizer.step(self._closure) 

    def _closure(self):

        self.optimizer.zero_grad()
        self.r = self.model(self.t) 
        self._compute_thrust()
        self.loss_physics = torch.mean(self.T.norm(dim=1)) * self.w_physics
        
        if self.w_bc > 0:
            self.loss_bc = (torch.mean((self.r[0] - self.r0)**2) + torch.mean((self.r[-1] - self.r1)**2)) / 2 * self.w_bc
            self.loss = self.loss_bc + self.loss_physics

            self.loss_physics_history.append(self.loss_physics.detach())
            self.loss_bc_history.append(self.loss_bc.detach())
            self.loss_history.append(self.loss.detach())
            self.loss.backward()
            return self.loss
        else:
            self.loss = self.loss_physics
            self.loss.backward()
            return self.loss_physics
       