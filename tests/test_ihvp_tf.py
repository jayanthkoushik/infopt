"""test_ihvp_tf.py: tests for inverse Hessian-vector product
in the TensorFlow modules."""

import shutil
import sys
import unittest
from typing import Optional

import tensorflow as tf

from infopt.ihvp_tf import IterativeIHVPTF, LowRankIHVPTF

CHECK_PLACES = 4  # Number of decimal places used for comparing numbers

TB_WRITER: Optional[tf.summary.SummaryWriter]
if sys.gettrace():
    tb_log_dir = "logs/test_ihvp_tf"
    try:
        shutil.rmtree(tb_log_dir)
    except FileNotFoundError:
        pass
    TB_WRITER = tf.summary.SummaryWriter(tb_log_dir)
else:
    TB_WRITER = None


class TestIHVPTF(unittest.TestCase):

    """Test cases for different IHVP implementations in TensorFlow."""

    def setUp(self):
        """Set up for a simple quadratic objective:

        f(x, y) = 0.5 * [x, y]^T * M * [x, y]."""

        # x and y are two parameter tensors
        x = tf.Variable(tf.random.normal(shape=(2, 2)))
        nx = x.shape.num_elements()
        y = tf.Variable(tf.random.normal(shape=(3, 2)))
        ny = y.shape.num_elements()
        n = nx + ny

        # get well-conditioned quadratic coefficients for the objective
        A = tf.random.normal(shape=(n, n))
        M = tf.transpose(A) @ A + 0.01 * tf.linalg.eye(n)
        E, V = tf.linalg.eigh(M)
        emin_good = E[-1] / 5
        E = tf.maximum(E, emin_good)  # E_max/E_min <= 0.2
        if E[-1] > 1:
            E = E / E[-1]
        self.M = V @ tf.linalg.diag(E) @ tf.transpose(V)
        self.M_inv = tf.linalg.inv(self.M)

        # build tf2 model & gradient tapes for gradient computations
        self._build_model(x, y)

        # get random vectors and true IHVPs on them
        self.vs, self.ihvp_true_flats = [], []
        for _ in range(10):
            v = [tf.random.normal(p.shape) for p in self.params]
            self.vs.append(v)
            v_flat = tf.expand_dims(tf.concat([tf.reshape(v_i, [-1])
                                               for v_i in v], -1), -1)
            ihvp_true_flat = self.M_inv @ v_flat
            self.ihvp_true_flats.append(ihvp_true_flat)

    def _build_model(self, x, y):
        """Build the model within inner & outer gradient tapes.

        Note that an active self.outer_tape is required for IHVP computations.
        """
        with tf.GradientTape(persistent=True) as self.outer_tape:
            with tf.GradientTape() as inner_tape:
                self.params = [x, y]
                x_flat = tf.reshape(x, [-1])
                y_flat = tf.reshape(y, [-1])
                self.z = tf.expand_dims(tf.concat([x_flat, y_flat], -1), 1)
                self.out = 0.5 * tf.reduce_sum(tf.multiply(self.z, self.M @ self.z))
            self.grad_params = inner_tape.gradient(self.out, self.params)

    def _test_ihvp(self, ihvp, simple=False):
        ihvp.update(self.out, self.grad_params, self.outer_tape, self.vs)
        for v, ihvp_true_flat in zip(self.vs, self.ihvp_true_flats):
            ihvp_app = ihvp.get_ihvp(v)
            self.assertIsInstance(ihvp_app, list, msg="IHVP is not a list")
            ihvp_app_flat = tf.expand_dims(tf.concat(
                [tf.reshape(ihvp_i, [-1]) for ihvp_i in ihvp_app],
                -1,
            ), -1)
            self.assertEqual(
                len(ihvp_app_flat),
                len(ihvp_true_flat),
                msg="Computed IHVP length does not match that of " "true value",
            )
            if simple:
                continue
            for q_true, q_app in zip(ihvp_true_flat, ihvp_app_flat):
                q_true, q_app = q_true.numpy().item(), q_app.numpy().item()
                self.assertAlmostEqual(
                    q_true,
                    q_app,
                    places=CHECK_PLACES,
                    msg="Computed IHVP does not " "match true value",
                )

    def test_iterative_ihvp(self):
        """Test IterativeIHVP."""
        ihvp = IterativeIHVPTF(self.params, iters=1000)
        self._test_ihvp(ihvp)
        ihvp.close()

    @unittest.expectedFailure
    def test_low_rank_ihvp(self):
        """Test LowRankIHVP."""
        ihvp = LowRankIHVPTF(
            self.params,
            rank=5,
            batch_size=5,
            iters_per_point=20,
            # device=DEVICE,
            tb_writer=TB_WRITER,
        )
        self._test_ihvp(ihvp)
        ihvp.close()

    def test_low_rank_ihvp_simple(self):
        """Test only elementary functionality of LowRankIHVP."""
        ihvp = LowRankIHVPTF(
            self.params,
            rank=3,
            batch_size=1,
            iters_per_point=100,
            # device=DEVICE,
            tb_writer=TB_WRITER,
        )
        self._test_ihvp(ihvp, simple=True)
        ihvp.close()


if __name__ == "__main__":
    unittest.main()
