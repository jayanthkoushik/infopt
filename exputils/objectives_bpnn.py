"""
Objectives for BPNN training (ZrO2 data, CdS data)
"""

import json
import logging
import os.path
import numpy as np
from tqdm import tqdm

from exputils.objectives import BaseObjective


class BPNNBandit(BaseObjective):
    """Find the molecular structure with the lowest calculated energy.

    Default data is the 2k ZrO2 dataset.

    This is cast as an armed bandit problem over a *categorical* space:
    each unexplored point corresponds to an arm.
    Note that the "true" inputs are a set of g-function atomic features, and
    the input size differs for different molecules.
    """
    def __init__(
            self,
            input_dim: int = 8,  # fixed
            search_split: float = 1.0,  # use some as initial pretraining points anyway
            n_test: int = 500,  # must not change within experiment!
            rng: np.random.Generator = np.random.default_rng(),
            json_path: str = "data/bpnn/data.json",
            gvecs_npy_path: str = "data/bpnn/gvecs.npy",
    ):
        super().__init__(input_dim)

        self.index2feature, self.Y = self.preprocess_or_load(json_path, gvecs_npy_path)
        self.n = len(self.Y)
        self.X = np.expand_dims(np.arange(self.n), 1)  # [[0], [1], ..., [n]]
        self.Y = self.Y / 1000  # kilo

        self.search_split = search_split
        self.n_test = n_test
        self.rng = (rng if isinstance(rng, np.random.Generator)
                    else np.random.default_rng(rng))
        self._split_data()

        # treat search options as categorical variables (no gradient descent)
        # note that the actual values are indices to X_search and Y_search
        self.domain = [{"name": "molecules",
                        "type": "categorical",
                        "domain": tuple(i for i in range(self.n_search))}]

        self.min = self.X_search[self.Y_search.argmin()]
        self.fmin = self.Y_search.min()

    @staticmethod
    def preprocess_or_load(json_path, gvecs_npy_path):
        """Either compute the input features using raw data (json) or
        retrieve pre-computed features from a .npy file."""
        if os.path.exists(gvecs_npy_path):
            data = np.load(gvecs_npy_path, allow_pickle=True)
        else:
            data = BPNNBandit.make_gfunction_data(json_path, gvecs_npy_path)
        X = data[:, :-1]  # each entry is either a list or a string 'O'
        Y = data[:, -1:].astype(float)
        return X, Y

    @staticmethod
    def make_gfunction_data(
            json_path,
            gvecs_npy_path,
            etas=(0.05, 4, 20, 80),
            cutoff=6.0):
        """Compute g-function features from raw molecular structures data.

        Results are stored in gvecs_npy_path.
        """
        with open(json_path, "r") as read_file:
            molecules = json.load(read_file)
        logging.info("processing raw molecules data from %s", read_file)

        cut_off = cutoff
        max_length = max(len(mol["numbers"]) for _, mol in molecules.items())
        logging.info("maximum number of atoms: %d", max_length)

        gvecs_for_cells = []
        for key in tqdm(molecules):
            all_positions = []
            all_positions_symbols = []

            cell = molecules[key]['cell']
            l = cell[0][0]
            h = cell[1][1]
            b = cell[2][2]

            positions = molecules[key]['positions']
            numbers = molecules[key]['numbers']

            for i in [0, l, -l]:
                for j in [0, h, -h]:
                    for k in [0, b, -b]:
                        for x, pos in enumerate(positions):
                            added = [round(pval + dval, 7) for pval, dval in zip(pos, [i, j, k])]
                            all_positions += [added]
                            all_positions_symbols += [molecules[key]["numbers"][x]]

            all_positions_set = [list(x) for x in set(tuple(x) for x in all_positions)]

            list_of_neighbors = []
            gvecs_for_cell = []

            for i in range(len(positions)):
                ip = []
                ip_neighbors = []
                ip_symbols = []
                my_symbol = numbers[i]
                my_position = positions[i]

                for j in range(len(all_positions_set)):
                    dist = np.linalg.norm(np.array(positions[i]) - np.array(all_positions_set[j]))
                    if dist != 0 and dist < cut_off:
                        ip.append(dist)
                        ip_neighbors.append(all_positions_set[j])
                        ip_symbols.append(all_positions_symbols[j])
                ip = np.array(ip)
                ip_symbols = np.array(ip_symbols)
                ip_neighbors = np.array(ip_neighbors)
                num_neighbors = len(ip_neighbors)

                gvec = []

                for eta in etas:
                    ridge = 0.0
                    ridge_d = 0.0
                    for count in range(num_neighbors):
                        symbol = ip_symbols[count]
                        R_j = ip_neighbors[count]
                        if symbol == my_symbol:
                            R_ij = np.linalg.norm(R_j - my_position)
                            ridge += (np.exp(-eta * (R_ij ** 2) / (cut_off ** 2)) * 0.5 * (
                                        np.cos(np.pi * R_ij / cut_off) + 1.))
                        else:
                            R_ij = np.linalg.norm(R_j - my_position)
                            ridge_d += (np.exp(-eta * (R_ij ** 2) / (cut_off ** 2)) * 0.5 * (
                                        np.cos(np.pi * R_ij / cut_off) + 1.))
                    gvec.append(ridge)
                    gvec.append(ridge_d)

                gvecs_for_cell.append(gvec)

                sum_of_neighbors = np.sum(ip)
                list_of_neighbors.append(str(sum_of_neighbors))

            len_of_this_cell = len(list_of_neighbors)
            list_of_neighbors += (max_length - len_of_this_cell) * [str(0)]
            gvecs_for_cell += (max_length - len_of_this_cell) * [str(0)]
            gvecs_for_cell.append(str(molecules[key]["energy"]))
            gvecs_for_cells.append(gvecs_for_cell)

        np.save(gvecs_npy_path, gvecs_for_cells, allow_pickle=True)
        logging.info("saved g-function features to %s", gvecs_npy_path)
        return gvecs_for_cells

    def _split_data(self):
        """Randomly split data into pretrain, search, and test portions.

        Note that the first n_search elements of X and Y (after shuffling)
        correspond to the search candidates.
        """
        # Test split is always fixed (after one shuffle)
        test_rng = np.random.default_rng(0)
        self.indices = test_rng.permutation(self.n)
        self.X, self.Y = self.X[self.indices], self.Y[self.indices]
        self.X_test, self.Y_test = self.X[-self.n_test:], self.Y[-self.n_test:]

        # Re-shuffle train/search split using self.rng
        self.indices[:-self.n_test] = self.rng.permutation(self.indices[:-self.n_test])
        self.n_search = int(self.search_split * (self.n - self.n_test))
        self.n_pretrain = self.n - self.n_search - self.n_test

        split0, split1 = self.n_search, -self.n_test
        self.X_search, self.Y_search = self.X[:split0], self.Y[:split0]
        self.X_pretrain, self.Y_pretrain = self.X[split0:split1], self.Y[split0:split1]

    def get_domain(self):
        """Get the domain set for the problem."""
        return self.domain

    def get_features(self, x):
        """Get atomic features given indices."""
        return self.index2feature[x.astype(int).squeeze(1)]

    def f(self, x):
        """Compute the output value given search indices."""
        return np.take(self.Y_search, np.array(x, np.int64))

    def f_noiseless(self, x):
        return self.f(x)

    def get_data(self, split="pretrain"):
        """Get input and output arrays for model training/evaluation."""
        X, Y = {
            "pretrain": (self.X_pretrain, self.Y_pretrain),
            "search": (self.X_search, self.Y_search),
            "test": (self.X_test, self.Y_test),
        }[split]
        X = self.index2feature[X.squeeze()]
        return X, Y

    def get_low_energy_indices(self, cutoff=-0.8, split="search"):
        """Retrieve the indices of low-energy candidates in a split,
        determined by a energy cutoff."""
        X, Y = {
            "pretrain": (self.X_pretrain, self.Y_pretrain),
            "search": (self.X_search, self.Y_search),
            "test": (self.X_test, self.Y_test),
        }[split]
        indices = np.squeeze(Y < cutoff)
        return X[indices]

    def compute_metrics(self, X_retrieved,
                        beta=1.0, cutoff=-0.8, split="search"):
        """Computes the precision, recall, and the F-beta score for retrieval.
        """
        X_true = set(np.unique(
            self.get_low_energy_indices(cutoff=cutoff, split=split).astype(int)
        ))
        X_retrieved = set(np.unique(X_retrieved).astype(int))
        precision = len(X_true & X_retrieved) / max(1, len(X_retrieved))
        recall = len(X_true & X_retrieved) / max(1, len(X_true))
        fbeta = (1 + beta ** 2) * (precision * recall)
        fbeta /= beta ** 2 * precision + recall + 1e-8
        return precision, recall, fbeta


class CdSBandit(BaseObjective):
    """Find low-energy CdS structures in a bandit setting.

    Makes use of the CdS dataset via
    [TensorMol](https://github.com/nguyenka/TensorMol).
    """
    def __init__(
            self,
            input_dim: int = 1120,
            setname: str = "",
    ):
        super().__init__(input_dim)

    def get_domain(self):
        """Get the domain set for the problem."""
        # return self.domain

    def get_features(self, x):
        """Get atomic features given indices."""
        # return self.index2feature[x.astype(int).squeeze(1)]

    def f(self, x):
        """Compute the output value given search indices."""
        # return np.take(self.Y_search, np.array(x, np.int64))

    def f_noiseless(self, x):
        # return self.f(x)
        pass
