"""ihvp_tf.py: inverse Hessian-vector product for TensorFlow neural nets.

HVP Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py
"""

from abc import ABC, abstractmethod
import random
from tqdm import tqdm

import numpy as np
import tensorflow as tf


class BaseIHVPTF(ABC):

    """Base class for computing Hessian-vector products in TensorFlow.

    Similar construction as the torch equivalent (`infopt.ihvp.BaseIHVP`),
    but importantly requires an active outer (second-order) gradient tape
    that is used to calculate `self.out`.
    """

    def __init__(self):
        self.out = None
        self.params = None
        self.grad_params = None
        self.outer_tape = None

    @abstractmethod
    def get_ihvp(self, v):
        """Compute [H_params(self.out)]^(-1)v."""
        raise NotImplementedError

    # pylint: disable=unused-argument
    def update(self, out, grad_params, outer_tape, vs):
        """Update the output, its gradients, and the outer gradient tape
        with respect to which ihvp is computed."""
        self.out = out
        self.grad_params = grad_params
        self.outer_tape = outer_tape
        assert self.outer_tape._persistent, \
            f"Second-order gradient tape must be persistent"

    def close(self):
        """Closes the outer gradient tape to free up memory.

        To be called after computing all ihvp's in the current training loop."""
        self.grad_params = None
        if self.outer_tape is None:
            return
        self.outer_tape._tape = None


class IterativeIHVPTF(BaseIHVPTF):

    """Combination of Taylor expansion with Perlmutter's method for
    comuting inverse Hessian-vector product approximation.

    It is important that the Hessian is positive definite, and has spectral
    norm not greater than 1.

    A TensorFlow counterpart to `infopt.ihvp.IterativeIHVP`.
    """

    def __init__(self, params, iters, reg=0.0, scale=1.0):
        """
        Arguments:
            params: list of parameter variables. The Hessian is computed wrt
                these.
            iters: iterations for approximation.
            reg: regularization added to the Hessian.
            scale: number by which the Hessian is divided.
        """
        super().__init__()
        self.params = list(params)
        self.n = len(self.params)
        self.iters = iters
        self.reg = reg
        self.scale = scale

    def compute_hvp(self, v):
        """Computes the HVP knowing gradients and their outer tape.

        v has the same shape as self.params and self.grad_params.
        """
        assert isinstance(v, list)
        assert len(v) == self.n
        assert self.grad_params is not None and self.outer_tape is not None
        # back-over-back hvp using the output_gradients option
        hvp = [tf.debugging.check_numerics(h_i, "hvp")
               for h_i in self.outer_tape.gradient(self.grad_params,
                                                   self.params,
                                                   output_gradients=v)]
        return hvp

    def get_ihvp(self, v):
        """Compute [H_params(self.out)]^(-1)v."""
        assert isinstance(v, list)
        assert len(v) == self.n
        assert self.grad_params is not None and self.outer_tape is not None
        ihvp = v[:]
        for _ in tqdm(range(self.iters), total=self.iters,
                      desc="IterativeIHVP_TF.get_ihvp"):
            # Apply the recursion ihvp <- v + ihvp - H*ihvp
            H_ihvp = self.compute_hvp(ihvp)
            for i in range(self.n):
                ihvp[i] = v[i] + (1.0 - self.reg) * ihvp[i] - H_ihvp[i]
                ihvp[i] = tf.stop_gradient(ihvp[i]) / self.scale
        return ihvp


class LowRankIHVPTF(BaseIHVPTF):

    """Approximate Hessian-vector products using a low rank approximation
    of the Hessian.

    The low rank matrix is itself represented using a neural network.

    Math:
    H ~ Q = P @ tr(P)
    P = U @ S @ tr(V)
    => H^{-1} ~ Q^{-1} = U @ (1/S**2) @ tr(U)

    AppNet(v) = P @ tr(P) @ v,
        where AppNet is an autoencoder and
        P is the weight matrix of AppNet's encoder.

    After training, compute U, S, V = svd(P), and obtain
        H^{-1}v ~ U @ (1/S**2) @ tr(U) @ v.
    We = U @ (1/S**2), Wt = tr(U).
    """

    @staticmethod
    def _make_app_net(n_ins, n_h):
        """Make the autoencoding NN that approximates Hv."""
        inputs = [tf.keras.Input(shape=n_in, name=f"input{i}")
                  for i, n_in in enumerate(n_ins)]
        fcs = [tf.keras.layers.Dense(n_h, name=f"fc{i}")
               for i, n_in in enumerate(n_ins)]
        hidden = tf.reduce_sum([fc(inp) for inp, fc in zip(inputs, fcs)], 0)
        ys = [hidden @ tf.transpose(fc.kernel) for fc in fcs]
        return tf.keras.Model(inputs=inputs, outputs=ys, name="app_net")

    class _AppNet(tf.keras.Model):
        """Make the autoencoding NN that approximates Hv."""

        def __init__(self, n_ins, n_h):
            super().__init__(self)
            self.n_ins = n_ins
            self.n_h = n_h
            self.fcs = [tf.keras.layers.Dense(n_h, name=f"fc{i}")
                        for i, n_in in enumerate(n_ins)]

        def call(self, inputs, training=None, mask=None):
            assert len(inputs) == len(self.n_ins)
            hidden = tf.reduce_sum([
                fc(inp) for inp, fc in zip(inputs, self.fcs)
            ], axis=0)
            ys = [tf.matmul(hidden, tf.transpose(fc.kernel)) for fc in self.fcs]
            return ys

        def train_step(self, data):
            raise NotImplementedError

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        params,
        rank,
        batch_size=np.inf,
        iters_per_point=20,
        criterion=tf.keras.losses.MeanSquaredError(),
        optim_cls=tf.keras.optimizers.Adam,
        optim_kwargs={"learning_rate": 0.01},
        ckpt_every=1,
        device="cpu",
        tb_writer=None,
    ):
        super().__init__()
        self.target_params = list(params)
        self.numels = [p.shape.num_elements() for p in self.target_params]
        self.criterion = criterion
        self.batch_size = batch_size
        self.iters_per_point = iters_per_point
        self.optim = optim_cls(**optim_kwargs)
        self.total_iter = 0
        self.n_train = 0
        self.ckpt_every = ckpt_every
        self.device = device
        self.tb_writer = tb_writer

        # Setup tf.keras model
        self.app_net = self._make_app_net(self.numels, rank)
        self.app_net.compile(self.optim, loss=self.criterion)

        self.P = tf.zeros(shape=(sum(self.numels), rank))
        self._update_wew()

    def _update_wew(self):
        """Update the helper tensors used for get_ihvp."""
        self.P = tf.concat([self.app_net.get_layer(f"fc{i}").kernel
                            for i in range(len(self.target_params))], axis=0)
        S, U, V = tf.linalg.svd(self.P)
        We = U / (S ** 2)
        Wt = tf.transpose(U)

        # Convert We and Wt to a list of tensors compatible with the params
        self.We, self.Wt = [], []
        n = 0
        for p, n_i in zip(self.target_params, self.numels):
            shape = list(p.shape)
            self.We.append(tf.reshape(We[n : n + n_i, :], shape + [-1]))
            self.Wt.append(tf.reshape(Wt[:, n : n + n_i], [-1] + shape))
            n += n_i

    def update(self, out, grad_params, outer_tape, vs):
        """Update the output, its gradients, and the outer gradient tape
        with respect to which ihvp is computed.

        vs (and their IHVPs) are used here as the training points for app_net.
        They have the same shape as self.params and self.grad_params.
        """
        self.out = out
        self.grad_params = grad_params
        self.outer_tape = outer_tape
        assert self.grad_params is not None and self.outer_tape is not None
        assert self.outer_tape._persistent, \
            f"Second-order gradient tape must be persistent"
        train_vs = vs
        # TODO(yj): batchify
        # for v in vs:
        #     # first dimension is batch size (1)
        #     train_vs.append([tf.reshape(v_i, [1, -1]) for v_i in v])
        n_new = max(1, len(train_vs) - self.n_train)
        self.n_train = len(train_vs)

        # grad_target_params = [tf.reshape(g, [-1]) for g in self.grad_params]

        # Step 1: Make batches of (v, hvp(v)) for training app_net
        idx = 0
        Y_train = []
        batch_size = min(self.batch_size, len(train_vs))
        while idx < self.n_train:
            batch_X = train_vs[idx : idx + batch_size]
            idx += len(batch_X)
            batch_Y = [
                # back-over-back hvp.
                tf.concat([
                    tf.reshape(hvp_i, [-1])
                    for hvp_i in self.outer_tape.gradient(self.grad_params,
                                                          self.target_params,
                                                          output_gradients=v)
                ], axis=0)
                for v in batch_X
            ]  # list of 1-d tensors of shape (#params, )
            Y_train.extend(batch_Y)

        # Step 2: Train app_net
        # self.app_net.fit(
        #     train_vs,
        #     Y_train,
        #     batch_size,
        #     (self.iters_per_point * n_new) // self.n_train,
        #     verbose=0,
        #     shuffle=True,
        # )
        iters = self.iters_per_point * n_new
        for _ in range(iters):
            idxs = random.sample(range(self.n_train), batch_size)
            batch_X = [[tf.reshape(v_i, [1, -1]) for v_i in train_vs[idx]]
                       for idx in idxs]
            batch_Y = [Y_train[idx] for idx in idxs]

            # Compute predictions and loss
            with tf.GradientTape() as tape:
                batch_Yhat = self.app_net([tf.concat(x, 0)
                                           for x in zip(*batch_X)])
                batch_Yhat = list(zip(*batch_Yhat))
                loss = sum(
                    self.criterion(y, tf.concat(y_hat, axis=0)) / batch_size
                    for y, y_hat in zip(batch_Y, batch_Yhat)
                )
            gradients = tape.gradient(loss, self.app_net.trainable_variables)
            self.optim.apply_gradients(zip(gradients,
                                           self.app_net.trainable_variables))
            self.total_iter += 1
            if (self.tb_writer is not None and
                self.total_iter % self.ckpt_every == 0):
                self.tb_writer.add_scalar(
                    "low_rank_ihvp/log_hv_loss",
                    np.log10(loss.numpy()).item(),
                    self.total_iter,
                )

        # Step 3: Update SVD of P
        self._update_wew()

    def get_ihvp(self, v):
        """Compute [H_params(self.out)]^(-1)v."""
        assert isinstance(v, list)
        assert len(v) == len(self.target_params)
        Wtv = sum([
            tf.reduce_sum(tf.reshape(Wt_i * tf.expand_dims(v_i, 0), [-1, n_i]),
                          axis=1)
            for Wt_i, v_i, n_i in zip(self.Wt, v, self.numels)
        ])  # (n_h, )
        Wtv = tf.expand_dims(Wtv, axis=1)  # (n_h, 1)
        iHv = [We_i @ Wtv for We_i in self.We]  # [(numel, )]
        return iHv

