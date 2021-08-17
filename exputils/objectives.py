"""objectives.py: objective functions to be tested."""

from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        # "bpnnbandit": BPNNBandit,
        "weatherbandit": WeatherBandit,
    }
    return objectives[name]


class BaseObjective:
    """Base object for optimization objectives.

    Args:
        input_dim (int): input dimension.

    Attributes:
        bounds (List[Tuple[float]]):
            list of lower and upper bounds for each input dimension.
        fmin (float): (approximate) minimum of the objective within bounds.
        min (np.ndarray): (approximate) minimizer of the objective.
        name (str): name of the objective.
    """

    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        self.bounds = [(-np.inf, np.inf) for _ in range(self.input_dim)]
        self.min = None
        self.fmin = None
        self.name = ""

    def f(self, x):
        raise NotImplementedError

    def f_noiseless(self, x):
        raise NotImplementedError


class Ackley(BaseObjective):
    """Ackley function.

    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    def __init__(self, input_dim, bounds=None, sd=None):
        super().__init__(input_dim)

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


class Rastrigin(BaseObjective):
    """Rastrigin function.

    This is a scaled version:

        f(x) = A + (1/n)*sum_{i=1}^n [x_i^2 - A*cos(2*π*x_i)]
    """

    def __init__(self, input_dim, A=10.0, bound=5.12):
        super().__init__(input_dim)
        self.bounds = [(-bound, bound)] * self.input_dim
        self.min = [0.0] * self.input_dim
        self.fmin = 0.0
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


class RandomNN(BaseObjective, nn.Module):
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
        super().__init__(input_dim)
        self.bounds = [(-bound, bound)] * input_dim
        self.min = []
        self.fmin = np.inf
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


class WeatherBandit(BaseObjective):
    """Find the location with the lowest temperature,
    given a set of locations and their (noisy) temperature measurements.

    Taken from:
        https://github.com/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_bandits_optimization.ipynb

    Attributes:
        input_dim (int): input dimension (fixed to 2)
        use_max (bool): whether to use the maximum or minimum temperature
        data_file (str): location of the weather data file
        coords (np.ndarray): array containing all (lon, lat) coordinates
        temps (np.ndarray): array containing the maximum/minimum temperatures
        bounds (list of np.ndarray): bounds on lon/lat
        min (np.ndarray): location of the maximum/minimum temperature
        fmin (float): value of the maximum/minimum temperature
    """
    def __init__(
            self,
            input_dim: int = 2,  # fixed
            use_max: bool = True,
            noise: float = 0.0,
            data_file: str = "data/weather/target_day_20140422.dat",
            days_out: int = 0,
    ):
        super().__init__(input_dim)
        assert self.input_dim == 2, \
            f"input dimension for this problem must be 2"
        self.use_max = use_max
        self.data_file = data_file
        self.days_out = days_out
        self.noise = noise

        self._read_data()

    def _read_data(self):
        """Read and process temperature data."""
        self.df = pd.read_csv(
            self.data_file,
            delimiter=" ",
            header=0,
            index_col=False,
            names=["lat", "lon", "days_out", "MaxT", "MinT"],
        )
        # select actual temperatures (days_out==0); remove Alaska & the islands
        self.df = self.df.query(
            f"days_out == {self.days_out}"
            f" & lat > 24 & lat < 51 & lon > -130 & lon < -65"
        )

        self.coords = self.df[["lon", "lat"]].to_numpy()
        self.temps = self.df["MaxT" if self.use_max else "MinT"].to_numpy()
        self.min = self.coords[self.temps.argmin()]
        self.fmin = self.temps.min()
        self.bounds = list(np.stack([self.coords.min(axis=0),
                                     self.coords.max(axis=0)]).T)

    def find_index(self, x):
        """Find the index corresponding to the input location x.

        TODO: what if x is outside the domain?
        """
        index, = np.where(np.all(self.coords == x, axis=1))
        return index

    def f(self, x):
        fx = self.temps[self.find_index(x)]
        if self.noise > 0.0:
            return fx + np.random.normal(0, self.noise, size=1)
        else:
            return fx

    def f_noiseless(self, x):
        return self.temps[self.find_index(x)]

    def get_domain(self):
        return [{"name": "stations", "type": "bandit", "domain": self.coords}]

    def plot_coords(self, plot_style="seaborn-colorblind", save_filename=None):
        """Plot the coordinates on the US map."""
        plt.style.use(plot_style)
        plt.figure(figsize=(12, 7))
        plt.scatter(self.coords[:, 0], self.coords[:, 1],
                    c='b', s=2, edgecolors='none')
        plt.title('US weather stations', size=25)
        plt.xlabel('Logitude', size=15)
        plt.ylabel('Latitude', size=15)
        plt.ylim((25, 50))
        plt.xlim((-128, -65))
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename)

    def plot_result(self, bo,
                    plot_style="seaborn-colorblind", save_filename=None):
        """Plot the result of a learned BO model on the US map."""
        plt.style.use(plot_style)
        plt.figure(figsize=(15, 7))
        jet = plt.cm.get_cmap('jet')
        sc = plt.scatter(self.coords[:, 0], self.coords[:, 1],
                         c=self.temps, vmin=0, vmax=35, cmap=jet, s=3,
                         edgecolors='none')
        cbar = plt.colorbar(sc, shrink=1)
        temp = "Max" if self.use_max else "Min"
        cbar.set_label(f'{temp}. Temperature')
        plt.plot(bo.x_opt[0], bo.x_opt[1], 'ko', markersize=10,
                 label='Best found')
        plt.plot(bo.X[:, 0], bo.X[:, 1], 'k.', markersize=8,
                 label='Observed stations')
        plt.plot(self.coords[np.argmin(self.temps), 0],
                 self.coords[np.argmin(self.temps), 1],
                 'k*', markersize=15, label='Coldest station')
        plt.legend()
        plt.ylim((25, 50))
        plt.xlim((-128, -65))
        plt.title(f'{temp}. temperature: April, 22, 2014', size=25)
        plt.xlabel('Longitude', size=15)
        plt.ylabel('Latitude', size=15)
        plt.text(-125, 28, 'Total stations =' + str(self.coords.shape[0]),
                 size=20)
        plt.text(-125, 26.5, 'Sampled stations =' + str(bo.X.shape[0]), size=20)
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename)

    def plot_histogram(self, bo,
                       plot_style="seaborn-colorblind", save_filename=None):
        """Plot a histogram of temperatures along with the best prediction."""
        plt.style.use(plot_style)
        plt.figure(figsize=(8, 5))
        plt.hist(self.temps, bins=50)
        temp = "Max" if self.use_max else "Min"
        plt.title(f'Distribution of {temp}. Temperatures', size=25)
        plt.vlines(self.fmin, 0, 1000, lw=3, label='Coldest station')
        plt.vlines(bo.fx_opt, 0, 1000, lw=3, linestyles=u'dotted',
                   label='Best found')
        plt.legend()
        plt.xlabel(f'{temp}. temperature', size=15)
        plt.xlabel('Frequency', size=15)
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename)