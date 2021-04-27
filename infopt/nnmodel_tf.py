"""nnmodel_tf.py: a TensorFlow neural network model for GPyOpt."""

import random
import numpy as np
import tensorflow as tf

from GPyOpt.models.base import BOModel


class NNModelTF(BOModel):
    """TensorFlow neural network for modeling the objective function.

    TODO(yj): device control via context
    TODO(yj): change tb_writer via context
    """

    def __init__(
        self,
        net,      # tf.keras.models.Model
        ihvp,     # infopt.nnmodel_tf.*
        optim,    # tf.keras.optimizers.Optimizer
        criterion=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        update_batch_size=np.inf,
        update_iters_per_point=25,
        num_higs=np.inf,
        ihvp_n=np.inf,
        ckpt_every=1,
        device="cpu",
        tb_writer=None,
    ):
        super().__init__()
        self.net = net
        self.criterion = criterion
        assert (hasattr(self.criterion, "reduction") and
                (self.criterion.reduction == tf.keras.losses.Reduction.NONE)), (
            f"criterion must have reduction=='none', "
            f"but got {self.criterion.reduction}"
        )
        self.ihvp = ihvp
        self.higs = []
        self.n = 0
        self.optim = optim
        self.update_batch_size = update_batch_size
        self.update_iters_per_point = update_iters_per_point
        self.num_higs = num_higs
        self.ihvp_n = ihvp_n
        self.ckpt_every = ckpt_every
        self.total_iter = 0
        self.device = device
        self.tb_writer = tb_writer
        self.dsdx = None

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """Train the NN using (X_all, Y_all) and get new IHVP estimates."""
        # Note: X_new and Y_new are ignored
        # pylint: disable=unused-argument
        X = tf.convert_to_tensor(X_all, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y_all, dtype=tf.float32)
        data = tf.data.Dataset.from_tensor_slices((X, Y))
        n_new = len(data) - self.n
        self.n = len(data)

        # Part 1: Train NN using current data points
        iters = self.update_iters_per_point * n_new
        batch_size = min(self.n, self.update_batch_size)

        # Change from torch: shuffle all data once before iterating through all
        shuffled_batches = data.shuffle(128).repeat().batch(batch_size)
        for i, (batch_X, batch_Y) in shuffled_batches.enumerate():
            # Standard TF2 train step
            with tf.GradientTape() as tape:
                batch_Yhat = self.net(batch_X, training=True)
                loss = tf.reduce_mean(
                    self.criterion(batch_Y[:, tf.newaxis],
                                   batch_Yhat[:, tf.newaxis])
                )
            gradients = tape.gradient(loss, self.net.trainable_variables)
            self.optim.apply_gradients(zip(gradients,
                                           self.net.trainable_variables))

            self.total_iter += 1
            if (self.tb_writer is not None and
                    self.total_iter % self.ckpt_every == 0):
                self.tb_writer.add_scalar(
                    "nn_model/log_loss", np.log10(loss.item()), self.total_iter
                )

            if i >= iters - 1:
                break

        # Part 2: Update IHVP calculators (IterativeIHVP or LowRankIHVP)
        ihvp_n = min(self.n, self.ihvp_n)
        idxs = random.sample(range(self.n), ihvp_n)
        X_idxs, Y_idxs = [tf.gather(t, idxs, axis=0) for t in [X, Y]]

        # TF2: an active outer gradient tape must be passed to ihvp
        #      for second-order gradient computations.
        with tf.GradientTape(persistent=True) as outer_tape:
            mean_loss, grad_params, dls = self._compute_jacobian(X_idxs, Y_idxs)
        # gradient of per-example loss w.r.t. parameters
        dls = list(zip(*[list(dl) for dl in dls]))
        assert len(dls) == ihvp_n
        assert len(dls[0]) == len(self.net.trainable_variables)

        # TF2: requires passing grad_params & outer_tape to ihvp
        self.ihvp.update(mean_loss, grad_params, outer_tape, dls)

        # Compute H^{-1}grad(L(z)) for selected points
        num_higs = min(len(dls), self.num_higs)
        dls = random.sample(dls, num_higs)
        self.higs = []
        for dl in dls:
            # grad(L(z)) needs to be "detached"
            dl0 = [tf.stop_gradient(tf.identity(g)) for g in dl]
            self.higs.append(self.ihvp.get_ihvp(dl0))

        # Close outer tape
        self.ihvp.close()

    @tf.function(experimental_relax_shapes=True)
    def _compute_jacobian(self, X_idxs, Y_idxs):
        with tf.GradientTape() as inner_tape:
            Yhat_idxs = self.net(X_idxs, training=True)
            losses = self.criterion(Y_idxs[:, tf.newaxis],
                                    Yhat_idxs[:, tf.newaxis])  # per example
        dls = inner_tape.jacobian(losses, self.net.trainable_variables)
        # gradient of mean loss w.r.t. parameters
        grad_params = [tf.reduce_mean(dl, axis=0) for dl in dls]
        # gradient of per-example loss w.r.t. parameters
        mean_loss = tf.reduce_mean(losses)
        return mean_loss, grad_params, dls

    def _predict_single(self, x, comp_grads=True):
        """A helper for predict() and predict_withGradients() per example."""

        x = tf.Variable(x)
        v = tf.constant(0.0, dtype=tf.float32)  # variance = sum(influence^2)
        with tf.GradientTape(persistent=True) as outer_tape:
            with tf.GradientTape() as inner_tape:
                m = self.net(x, training=False)
            grads = inner_tape.gradient(m, self.net.trainable_variables + [x])
            dmdp, dmdx = grads[:-1], grads[-1]
            if comp_grads:
                self.dmdx = dmdx
                self.dsdx = tf.zeros_like(x, dtype=tf.float32)

            # Calculate s and dsdx
            # hig: H^{-1}grad(L(z))
            # influence: sum(dmdp * H^{-1}grad(L(z)))
            dmdp_higs = [
                [tf.reduce_sum(tf.multiply(dmdp_i, tf.squeeze(hig_i)))
                 for dmdp_i, hig_i in zip(dmdp, hig)]
                for hig in self.higs
            ]
            influences = [sum(dmdp_hig) for dmdp_hig in dmdp_higs]
            v = tf.reduce_mean(tf.square(influences))

        if comp_grads:
            dmdpdx_higs = outer_tape.gradient(dmdp_higs, x)
            self.dsdx = tf.reduce_mean([
                influence * dmdpdx_hig
                for influence, dmdpdx_hig in zip(influences, dmdpdx_higs)
            ])

        s = tf.math.sqrt(v)
        if comp_grads:
            if s > 0:
                self.dsdx /= s
        else:
            self.dmdx = None
            self.dsdx = None

        return m, s, self.dmdx, self.dsdx

    def predict(self, X):
        """Get the predicted mean and std at X."""
        M, S = [], []
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(len(X)):
            x = X[i : i + 1]
            m, s, _, _ = self._predict_single(x, comp_grads=False)
            M.append(m)
            S.append(s.numpy().item())
        M = tf.concat(M, 0).numpy()
        S = np.array(S)[:, np.newaxis]
        return M, S

    def predict_withGradients(self, X):
        """Get the gradients of the predicted mean and variance at X."""
        M, S, dM, dS = [], [], [], []
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(len(X)):
            x = X[i : i + 1]
            m, s, dm, ds = self._predict_single(x, comp_grads=True)
            M.append(m)
            S.append(s.numpy().item())
            dM.append(dm)
            dS.append(ds)
        M = tf.concat(M, 0).numpy()
        S = np.array(S)[:, np.newaxis]
        dM = tf.concat(dM, 0).numpy()
        dS = tf.concat(dS, 0).numpy()
        return M, S, dM, dS

    def get_fmin(self):
        raise NotImplementedError

    def get_model_parameters(self):
        """Get parameters to be saved."""
        # pylint: disable=no-self-use
        return []
