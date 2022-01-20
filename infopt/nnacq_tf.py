"""nnacq_tf.py: LCB acquisition function (and optimizer)
for TensorFlow neural network models.

WARNING:tensorflow:11 out of the last 11 calls to <function pfor.<locals>.f at 0x173aded40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
"""

import numpy as np
import tensorflow as tf

from GPyOpt.acquisitions import AcquisitionLCB
from GPyOpt.experiment_design import RandomDesign

from infopt import utils


class NNAcqTF(AcquisitionLCB):

    """LCB acquisition for TensorFlow neural network model.

    The difference is the use of a gradient descent optimizer.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        model,
        space,
        exploration_weight=2,
        optim_cls=tf.keras.optimizers.Adam,
        optim_kwargs={"learning_rate": 0.01},
        optim_iters=100,
        optim_ckpt_every=1,
        fast_project=None,
        rel_tol=1e-4,
        device="cpu",
        tb_writer=None,
        reinit_optim_start=False,
        lr_decay_step_size=10,
        lr_decay_gamma=0.5,
        x_sampler=None,
    ):
        super().__init__(model, space, exploration_weight=exploration_weight)
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.optim_iters = optim_iters
        self.optim_ckpt_every = optim_ckpt_every
        self.fast_project = fast_project
        self.rel_tol = rel_tol
        self.device = device
        self.tb_writer = tb_writer
        self.reinit_optim_start = reinit_optim_start
        self.x_sampler = x_sampler
        if x_sampler is None:
            self.random_design = RandomDesign(space)
        self.lr_decay_step_size = lr_decay_step_size
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = self
        self.acq_calls = 0
        self.x = None
        self.optim = None
        self.lr_scheduler = None
        self._init_x()
        self.global_iter = 0

    def _init_x(self):
        if self.x_sampler is None:
            x = self.random_design.get_samples(init_points_count=1)
            x_data = tf.convert_to_tensor(x, dtype=tf.float32)
        else:
            x_data = self.x_sampler()

        if self.x is None:
            self.x = tf.Variable(x_data)
        else:
            self.x.assign(x_data)
        # not natural, but keep arguments consistent with the pytorch equivalent
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            self.optim_kwargs["learning_rate"],
            decay_steps=self.lr_decay_step_size,
            decay_rate=self.lr_decay_gamma,
            staircase=True,
        )
        tf2_optim_kwargs = {k: v for k, v in self.optim_kwargs.items()
                            if k != "learning_rate"}
        self.optim = self.optim_cls(learning_rate=self.lr_scheduler,
                                    **tf2_optim_kwargs)

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def optimize(self, duplicate_manager=None):
        # pylint: disable=protected-access, unused-argument
        """Override default optimizer to a gradient descent optimizer on x."""
        if self.reinit_optim_start:
            self._init_x()

        fx = float("inf")
        for iter_ in range(self.optim_iters):
            m, s, dmdx, dsdx = self.model._predict_single(self.x,
                                                          comp_grads=True)
            fx, old_fx = (m - self.exploration_weight * s).numpy().item(), fx
            dfx = dmdx - self.exploration_weight * dsdx
            self.optim.apply_gradients([(dfx, self.x)])

            if self.tb_writer is not None:
                if (
                    (iter_ + 1) % self.optim_ckpt_every == 0
                ) or iter_ == self.optim_iters - 1:
                    self.tb_writer.add_scalar(
                        "nn_model/acq_optim", fx, self.global_iter
                    )

            # Project point to space
            if self.fast_project is not None:
                self.fast_project(self.x.value())
            else:
                xdata_np = self.x.numpy()
                xdata_round_np = self.space.round_optimum(xdata_np)
                x_round = tf.convert_to_tensor(xdata_round_np, dtype=tf.float32)
                self.x.assign(x_round)

            self.global_iter += 1

            # Check tolerance criterion
            if abs(fx - old_fx) < abs(old_fx) * self.rel_tol:
                break

        m, s, _, _ = self.model._predict_single(self.x, comp_grads=False)
        fx = (m - self.exploration_weight * s).numpy().item()
        self.acq_calls += 1
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("nn_model/acq", fx, self.acq_calls)
        return self.x.numpy(), fx


class NNAcqCategoricalTF(AcquisitionLCB):
    """The LCB (or greedy, if exploration_weight == 0) optimizer defined
    on a categorical space.

    Resorts to an argmin over the input space, with sampling.
    """

    # no gradient-based optimization
    # (any _withGradients methods shouldn't be called)
    analytical_gradient_prediction = False

    def __init__(
            self,
            model,
            space,
            exploration_weight=2,
            n_candidates=np.inf,
            batch_size=None,  # TF default: 32
            tb_writer=None,
            reinit_optim_start=False,
            x_sampler=None,
            feature_map=None,
    ):
        super().__init__(model, space, exploration_weight=exploration_weight)

        self._check_space()
        self.domain = self.space.space_expanded[0].domain
        self.n_inputs = len(self.domain)

        self.n_candidates = min(n_candidates, self.n_inputs)
        self.batch_size = batch_size
        self.tb_writer = tb_writer
        self.reinit_optim_start = reinit_optim_start
        self.x_sampler = x_sampler
        self.feature_map = feature_map

        self.candidates = None
        self.candidates_features = None
        self._init_candidates()

        self.optimizer = self
        self.acq_calls = 0

    def _check_space(self):
        """Check if input space contains exactly one categorical variable."""
        assert len(self.space.space_expanded) == 1, (
            "input space must contain exactly one variable for " +
            self.__class__.__name__
        )
        assert self.space.space_expanded[0].type == "categorical", (
            "input variable must be categorical for " + self.__class__.__name__
        )

    def _init_candidates(self):
        """Samples/Retrieves all inputs to be considered for the argmin."""
        if self.x_sampler is None:
            if self.n_candidates < self.n_inputs:
                x = np.random.choice(self.domain,
                                     self.n_candidates,
                                     replace=False)[:, np.newaxis]
            else:
                x = np.arange(self.n_inputs)[:, np.newaxis]
        else:
            # x_sampler must be callable with n_candidates as an argument
            x = self.x_sampler(self.n_candidates)

        # use one-hot representation
        self.candidates = np.zeros((self.n_candidates, self.n_inputs))
        np.put_along_axis(self.candidates, x, 1, axis=1)

        if self.feature_map is None:
            x_data = tf.convert_to_tensor(self.candidates, dtype=tf.float32)
        else:
            x_data, _ = self.feature_map(self.candidates, None)
        self.candidates_features = x_data

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def optimize(self, duplicate_manager=None):
        # pylint: disable=protected-access, unused-argument
        """Override default optimizer to a gradient descent optimizer on x."""
        if self.reinit_optim_start:
            self._init_candidates()

        # loops over each x; slow
        if self.exploration_weight > 0:
            m, s = self.model.predict(self.candidates)
            assert len(m.shape) == 2 and m.shape[1] == s.shape[1] == 1, (
                f"m.shape: {m.shape}, s.shape: {s.shape}"
            )
        # greedy w/ batched prediction (tf.keras.Model)
        else:
            m = np.atleast_2d(utils.normalize_output(
                self.model.net.predict(self.candidates_features,
                                       self.batch_size)
            ))
            s = 0
        lcbs = m - self.exploration_weight * s  # N x 1

        fx = lcbs.min()
        min_idx = lcbs.argmin(axis=0)
        x = self.candidates[min_idx]
        # print("Acquisition:", np.argmax(x), ", LCB:", fx)

        self.acq_calls += 1
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("nn_model/acq", fx, self.acq_calls)
            self.tb_writer.add_scalar("nn_model/exp_w",
                                      self.exploration_weight, self.acq_calls)
            mu_min = m[min_idx].item()
            sig_min = s[min_idx].item() if self.exploration_weight > 0 else 0
            self.tb_writer.add_scalar("nn_model/mu", mu_min, self.acq_calls)
            self.tb_writer.add_scalar("nn_model/sigma", sig_min, self.acq_calls)

        # must return the one-hot vector as the minimizer
        return x, fx
