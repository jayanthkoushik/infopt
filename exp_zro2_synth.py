"""exp_zro2_synth.py: predicting energy of ZrO2 configurations."""

import abc
from collections import namedtuple
import itertools

import numpy as np
from scipy.spatial.distance import cdist

Atom = namedtuple("Atom", ["no", "position"])
"""An atom represented as (atomic number, position)."""


class MoleculeStructure:

    """Data structure representing a single molecule."""

    def __init__(self, cell, energy, forces, atoms):
        self.cell = cell
        self.energy = energy
        self.forces = forces
        self.atoms = atoms

        # Compute all positions
        self.all_atoms = []
        for a in atoms:
            for c in itertools.product(
                [0, cell[0][0], -cell[0][0]],
                [0, cell[1][1], -cell[1][1]],
                [0, cell[2][2], -cell[2][2]],
            ):
                p_c = [p_i + c_i for p_i, c_i in zip(a.position, c)]
                a_c = Atom(a.no, p_c)
                self.all_atoms.append(a_c)

    @staticmethod
    def from_json(mdata):
        cell = mdata["cell"]
        energy = mdata["energy"]
        forces = mdata["forces"]
        atoms = [Atom(n, p) for n, p in zip(mdata["numbers"], mdata["positions"])]
        return MoleculeStructure(cell, energy, forces, atoms)


class SymmetryFunction(abc.ABC):

    """Base class for symmetry functions."""

    def __init__(self, cutoff_dist):
        self.cutoff_dist = cutoff_dist

    def _get_neighbors(self, mol):
        """Get neighbors for atoms in the molecule."""
        dists = cdist(
            [a.position for a in mol.atoms], [a.position for a in mol.all_atoms]
        )
        dists_thresh = dists < self.cutoff_dist
        neighbors, ndists = [], []
        for i, d_i in enumerate(dists_thresh):
            i_neighbors = np.nonzero(d_i)[0]
            neighbors.append([mol.all_atoms[j] for j in i_neighbors if j != i])
            ndists.append([d_i[j] for j in i_neighbors if j != i])
        return neighbors, ndists

    @abc.abstractmethod
    def __call__(self, mol):
        pass


class AngularSymmetryFunction(SymmetryFunction):
    def __init__(self, eta, cutoff_dist):
        super().__init__(cutoff_dist)
        self.eta = eta

    def __call__(self, mol):
        _, ndists = self._get_neighbors(mol)
        G = []
        for d_i in ndists:
            G_i = 0.0
            for Rij in d_i:
                G_i += (
                    np.exp(-self.eta * Rij * Rij)
                    * 0.5
                    * (np.cos(np.pi * Rij / self.cutoff_dist) + 1)
                )
            G.append(G_i)
        return G
