"""test_ihvp.py: tests for inverse Hessian-vector product."""

import shutil
import sys
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter

from infopt.ihvp import IterativeIHVP, LowRankIHVP

CHECK_PLACES = 4  # Number of decimal places used for comparing numbers

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

if sys.gettrace():
    tb_log_dir = "logs/test_ihvp"
    try:
        shutil.rmtree(tb_log_dir)
    except FileNotFoundError:
        pass
    TB_WRITER = SummaryWriter(tb_log_dir)
else:
    TB_WRITER = None


class TestIHVP(unittest.TestCase):

    """Test cases for different IHVP implementations."""

    def setUp(self):
        """Set up for a simple quadratic objective."""

        x = torch.randn(2, 2, requires_grad=True, device=DEVICE)
        nx = x.numel()
        y = torch.randn(3, 2, requires_grad=True, device=DEVICE)
        ny = y.numel()
        n = nx + ny

        self.params = [x, y]
        x_flat, y_flat = x.view(nx), y.view(ny)
        self.z = torch.cat([x_flat, y_flat])

        A = torch.randn(n, n, device=DEVICE)
        M = A.t() @ A + 0.01 * torch.eye(n, device=DEVICE)
        E, V = torch.symeig(M, eigenvectors=True)
        # Get a well conditioned matrix
        emin_good = E[-1] / 5
        E[E < emin_good] = emin_good
        if E[-1] > 1:
            E = E / E[-1]
        self.M = V @ torch.diag(E) @ V.t()
        self.M_inv = torch.inverse(self.M)
        self.out = 0.5 * torch.sum(self.z * (self.M @ self.z))

        self.vs, self.ihvp_true_flats = [], []
        for _ in range(10):
            v = [torch.randn(p.shape, device=DEVICE) for p in self.params]
            self.vs.append(v)
            v_flat = torch.cat([v_i.view(v_i.numel()) for v_i in v])
            ihvp_true_flat = self.M_inv @ v_flat
            self.ihvp_true_flats.append(ihvp_true_flat)

    def _test_ihvp(self, ihvp, simple=False):
        ihvp.update(self.out, self.vs)
        for v, ihvp_true_flat in zip(self.vs, self.ihvp_true_flats):
            ihvp_app = ihvp.get_ihvp(v)
            self.assertIsInstance(ihvp_app, list, msg="IHVP is not a list")
            ihvp_app_flat = torch.cat(
                [ihvp_i.view(ihvp_i.numel()) for ihvp_i in ihvp_app]
            )
            self.assertEqual(
                len(ihvp_app_flat),
                len(ihvp_true_flat),
                msg="Computed IHVP length does not match that of " "true value",
            )
            if simple:
                continue
            for q_true, q_app in zip(ihvp_true_flat, ihvp_app_flat):
                q_true, q_app = q_true.item(), q_app.item()
                self.assertAlmostEqual(
                    q_true,
                    q_app,
                    places=CHECK_PLACES,
                    msg="Computed IHVP does not " "match true value",
                )

    def test_iterative_ihvp(self):
        """Test IterativeIHVP."""
        ihvp = IterativeIHVP(self.params, iters=1000)
        self._test_ihvp(ihvp)

    @unittest.expectedFailure
    def test_low_rank_ihvp(self):
        """Test LowRankIHVP."""
        ihvp = LowRankIHVP(
            self.params,
            rank=5,
            batch_size=5,
            iters_per_point=20,
            device=DEVICE,
            tb_writer=TB_WRITER,
        )
        self._test_ihvp(ihvp)

    def test_low_rank_ihvp_simple(self):
        """Test only elementary functionality of LowRankIHVP."""
        ihvp = LowRankIHVP(
            self.params,
            rank=3,
            batch_size=1,
            iters_per_point=100,
            device=DEVICE,
            tb_writer=TB_WRITER,
        )
        self._test_ihvp(ihvp, simple=True)
