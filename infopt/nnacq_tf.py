"""nnacq_tf.py: LCB acquisition function (and optimizer)
for TensorFlow neural network models.

WARNING:tensorflow:11 out of the last 11 calls to <function pfor.<locals>.f at 0x173aded40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
"""

import tensorflow as tf

from GPyOpt.acquisitions import AcquisitionLCB
from GPyOpt.experiment_design import RandomDesign


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
        """Override default optimizer to a gradient descent optimizer on x.

        TODO(yj): fix retracing issue
        """
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
