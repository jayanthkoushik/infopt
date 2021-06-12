"""objectives.py: objective functions to be tested."""

import numpy as np
from tqdm import tqdm
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import GPyOpt


def get_objective(name):
    name = name.lower()
    objectives = {
        "ackley": Ackley,
        "rastrigin": Rastrigin,
        "randomnn": RandomNN,
    }
    return objectives[name]


class Ackley:
    """
    Ackley function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim

        if bounds is None:
            self.bounds = [(-32.768, 32.768)] * self.input_dim
        else:
            self.bounds = bounds

        self.min = [0.] * self.input_dim
        self.fmin = 0

        if sd == None:
            self.sd = 0
        else:
            self.sd = sd

        self.name = f"{self.__class__.__name__} {self.input_dim}D"

    def f(self, X):
        X = GPyOpt.util.general.reshape(X, self.input_dim)

        n = X.shape[0]
        fval = (20 + np.exp(1) - 20 * np.exp(-0.2 * np.sqrt((X ** 2).sum(1) / self.input_dim)) - np.exp(
            np.cos(2 * np.pi * X).sum(1) / self.input_dim))

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        return fval.reshape(n, 1) + noise

    def f_noiseless(self, X):
        return self.f(X)


class Rastrigin:
    """Rastrigin function.

    This is a scaled version:

        f(x) = A + (1/n)*sum_{i=1}^n [x_i^2 - A*cos(2*π*x_i)]
    """

    def __init__(self, n, A=10.0, bound=5.12):
        self.bounds = [(-bound, bound)] * n
        self.min = [0.0] * n
        self.fmin = 0.0
        self.input_dim = n
        self.A = A
        self.range = [0, 2 * self.A + bound * bound]
        self.name = f"{self.__class__.__name__} {self.input_dim}D"

    def f(self, X):
        X = GPyOpt.util.general.reshape(X, self.input_dim)
        return self.A + np.mean(
            X ** 2 - self.A * np.cos(2 * np.pi * X), axis=1, keepdims=True
        )

    def f_noiseless(self, X):
        return self.f(X)


class RandomNN(nn.Module):
    """A randomly initialized fully-connected neural network
    used as an objective function."""

    def __init__(
            self,
            input_dim: int,
            layer_sizes: List[int],
            activation: str = "relu",
            bound: float = 10.,
            initializer=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.bounds = [(-bound, bound)] * input_dim
        self.min = []
        self.fmin = np.inf
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.layers = nn.Sequential(*[
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip([input_dim] + layer_sizes,
                                   layer_sizes + [1])
        ])
        self.activation = getattr(F, activation)

        if initializer is not None:
            self.initializer = initializer
        else:
            self.initializer = (
                lambda t: nn.init.kaiming_uniform_(t, a=np.sqrt(5))
            )

        for layer in self.layers:
            self.initializer(layer.weight)

        self.device = device
        self.to(device)

        self.name = f"{self.__class__.__name__}-{self.input_dim}d"
        self.name += f"-{layer_sizes}-{activation}"

    def find_fmin(self, n_mesh=100):
        """Find minimizer of the random neural network via grid search.

        This is the preferred method when input_dim is small.
        """
        grid = torch.meshgrid(*[torch.linspace(*bounds, n_mesh)
                                for bounds in self.bounds])
        X = torch.stack(grid, dim=-1).reshape(-1, self.input_dim)
        with torch.no_grad():
            Y = self(X)
            min_idx = Y.argmin()
            self.min = X[min_idx].detach().cpu().numpy()
            self.fmin = Y[min_idx].item()
        return X, Y

    def find_fmin_optim(self, optimizer_cls=torch.optim.LBFGS, max_iters=1000):
        """Find minimizer of the random neural network using an optimizer.

        Defaults to L-BFGS.
        """
        x = torch.zeros(self.input_dim, requires_grad=True)
        optimizer = optimizer_cls([x])
        for i in tqdm(range(max_iters), total=max_iters,
                      desc="finding minimizer of RandomNN"):

            def _closure():
                """Called multiple times in LBFGS."""
                x.data.clamp_(*self.bounds[0])
                optimizer.zero_grad()
                f = self(x)
                f.backward()
                return f

            optimizer.step(_closure)

            x.data.clamp_(*self.bounds[0])
            fval = self(x).item()
            if fval < self.fmin:
                self.min = x.detach().cpu().numpy()
                self.fmin = fval
                print(f"found new minimum in iteration {i}: {self.fmin:.2f}")

    def forward(self, X):
        h = X
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        out = self.layers[-1](h)
        return out

    def f(self, X):
        X_torch = torch.from_numpy(X).to(device=self.device,
                                         dtype=torch.float32)
        out = self.forward(X_torch)
        return out.detach().cpu().numpy()

    def f_noiseless(self, X):
        return self.f(X)

