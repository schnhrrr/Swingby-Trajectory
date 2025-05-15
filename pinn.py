import torch.nn as nn

class PINN(nn.Module):
    """
    A simple feedforward neural network for physics-informed neural networks (PINNs).
    The networks output can be transformed using a custom function.
    Args:
        N_INPUT (int): Number of input features.
        N_OUTPUT (int): Number of output features.
        N_NEURONS (int): Number of neurons in each hidden layer.
        N_LAYERS (int): Number of hidden layers.
        transform_fn (callable, optional): A custom function to transform the output. Defaults to None.

    """
    def __init__(self, N_INPUT, N_OUTPUT, N_NEURONS, N_LAYERS, transform_fn=None):
        super().__init__()
        self.activation = nn.Tanh
        self.transform_fn = transform_fn  # custom function
        self.fci = nn.Sequential(nn.Linear(N_INPUT, N_NEURONS), self.activation())
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(N_NEURONS, N_NEURONS), self.activation()) 
            for _ in range(N_LAYERS - 1)
        ])
        self.fco = nn.Linear(N_NEURONS, N_OUTPUT)

    def forward(self, t):
        # TODO: add another transform at beginning for fourier embedding
        x = self.fci(t)
        x = self.fch(x)
        x = self.fco(x)
        if self.transform_fn:
            return self.transform_fn(t, x)
        return x