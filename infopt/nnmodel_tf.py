"""nnmodel_tf.py: a TensorFlow neural network model for GPyOpt."""

import logging
import time
import random
import numpy as np
import tensorflow as tf

from GPyOpt.models.base import BOModel

import uncertainty_toolbox as uct

from infopt import utils
from infopt.recalibration import get_recalibrator


class NNModelTF(BOModel):
    """TensorFlow neural network for modeling the objective function."""

    MCMC_sampler = False
    analytical_gradient_prediction = True

    def __init__(
        self,
        net,      # tf.keras.models.Model
        ihvp,     # infopt.nnmodel_tf.*
        optim,    # tf.keras.optimizers.Optimizer
        criterion=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        update_batch_size=np.inf,
        update_iters_per_point=25,
        update_max_iters=np.inf,
        update_upsample_new=None,
        early_stopping=False,
        num_higs=np.inf,
        ihvp_n=np.inf,
        weight_decay=1e-5,
        recal_mode=None,
        recal_setsize=500,
        recal_kwargs=None,
        ckpt_every=1,
        device="cpu",
        tb_writer=None,
        **kwargs
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
        self.update_max_iters = update_max_iters
        self.update_upsample_new = update_upsample_new
        self.early_stopping = early_stopping
        self.num_higs = num_higs
        self.ihvp_n = ihvp_n
        self.weight_decay = weight_decay
        self.recal_mode = recal_mode
        self.recal_setsize = recal_setsize
        self.recal_kwargs = recal_kwargs
        self.recalibrator = None
        self.ckpt_every = ckpt_every
        self.total_iter = 0
        self.device = device
        self.tb_writer = tb_writer
        self.dsdx = None

        # idx -> features for categorical selection
        self.feature_map = kwargs.get("feature_map", None)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """Train the NN using (X_all, Y_all) and get new IHVP estimates."""
        # Note: X_new and Y_new are ignored (inputs: None, None)
        # pylint: disable=unused-argument

        # Leave a held-out calibration set as part of the initial points
        if self.recal_mode is not None:
            assert self.recal_setsize < len(Y_all), (
                "recalibration set size {} "
                "must be smaller than init points {}"
            ).format(self.recal_setsize, len(Y_all))
            X_recal = X_all[:self.recal_setsize]
            Y_recal = Y_all[:self.recal_setsize]
            X_all = X_all[self.recal_setsize:]
            Y_all = Y_all[self.recal_setsize:]
            logging.info("model update set: %d, recalibration set: %d",
                         len(Y_all), len(Y_recal))
        else:
            X_recal = None
            Y_recal = None

        if self.feature_map is not None:
            X, Y = self.feature_map(X_all, Y_all)
        else:
            X = tf.convert_to_tensor(X_all, dtype=tf.float32)
            Y = tf.convert_to_tensor(Y_all, dtype=tf.float32)

        # ratio of new samples in the training data
        if self.update_upsample_new is not None and self.update_upsample_new > 0.0:
            old_data = tf.data.Dataset.from_tensor_slices(
                (utils.slice_tensors(X, end=self.n),
                 utils.slice_tensors(Y, end=self.n))
            )
            n_new = len(Y) - self.n
            new_data = tf.data.Dataset.from_tensor_slices(
                (utils.slice_tensors(X, start=self.n),
                 utils.slice_tensors(Y, start=self.n))
            )
            self.n = len(old_data) + len(new_data)

            p = self.update_upsample_new
            new_to_old = min(p / (1 - p + 1e-8), 100)
            repeat_new = int(np.round(max(1, new_to_old * self.n / n_new)))
            data = new_data.repeat(repeat_new).concatenate(old_data)
        else:
            data = tf.data.Dataset.from_tensor_slices((X, Y))
            n_new = len(data) - self.n
            self.n = len(data)

        # Part 1: Train NN using current data points
        iters = min(self.update_iters_per_point * n_new, self.update_max_iters)
        batch_size = min(self.n, self.update_batch_size)

        # Change from torch: shuffle all data once before iterating through all
        old_loss = 1e-8
        shuffled_batches = data.shuffle(128).repeat().batch(batch_size)
        t0 = time.time()
        for i, (batch_X, batch_Y) in shuffled_batches.enumerate():
            # Standard TF2 train step
            with tf.GradientTape() as tape:
                batch_Yhat = utils.normalize_output(
                    self.net(batch_X, training=True))
                loss = tf.reduce_mean(
                    self.criterion(batch_Y[:, tf.newaxis],
                                   batch_Yhat[:, tf.newaxis])
                )
                if self.weight_decay > 0.0:
                    loss += self.weight_decay * tf.reduce_sum([
                        tf.reduce_sum(v ** 2)
                        for v in self.net.trainable_variables
                    ])
            gradients = tape.gradient(loss, self.net.trainable_variables)
            self.optim.apply_gradients(zip(gradients,
                                           self.net.trainable_variables))

            loss = loss.numpy().item()
            self.total_iter += 1
            if self.total_iter % self.ckpt_every == 0:
                time_per_iter = (time.time() - t0) / self.ckpt_every
                t0 = time.time()
                try:
                    lr = self.optim.lr.numpy()
                except AttributeError:
                    lr = self.optim.lr(self.optim.iterations).numpy()
                # logging.info("model update %5d (%5d/%5d) --- "
                #              "loss %.4f, lr %.4f, %.4f s/it",
                #              self.total_iter, i + 1, iters,
                #              loss, lr, time_per_iter)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        "nn_model/log_loss", np.log10(loss), self.total_iter,
                    )
                    self.tb_writer.add_scalar(
                        "nn_model/learning_rate", lr, self.total_iter,
                    )
            if (self.early_stopping and
                    abs(loss - old_loss) < abs(old_loss) * 1e-4):
                logging.info(f"loss converged after %d updates at loss %.5f",
                             i + 1, loss)
                break
            elif i >= iters - 1:
                logging.info(f"reached %d iterations with loss %.5f",
                             iters, loss)
                break
            else:
                old_loss = loss

        # Part 2: Update IHVP calculators (IterativeIHVP or LowRankIHVP)
        ihvp_n = min(self.n, self.ihvp_n)
        idxs = random.sample(range(self.n), ihvp_n)
        X_idxs, Y_idxs = [utils.gather_tensors(t, idxs, axis=0) for t in [X, Y]]

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

        # Part 3 (optional): update recalibrator & compute calibration error
        if self.recal_mode is not None:
            Y_pred, Y_std = self.update_recalibrator(X_recal, Y_recal)
            y_pred, y_std, y_recal = [y.squeeze(1)
                                      for y in [Y_pred, Y_std, Y_recal]]
            ece_uncal = uct.metrics_calibration.mean_absolute_calibration_error(
                y_pred, y_std, y_recal,
            )
            Y_std = self.recalibrate(Y_pred, Y_std)
            y_std = Y_std.squeeze(1)
            ece_recal = uct.metrics_calibration.mean_absolute_calibration_error(
                y_pred, y_std, y_recal,
            )
            logging.info("Iteration %d, recalibration: ECE %.5f -> %.5f",
                         self.total_iter, ece_uncal, ece_recal)

    @tf.function(experimental_relax_shapes=True)
    def _compute_jacobian(self, X_idxs, Y_idxs):
        with tf.GradientTape() as inner_tape:
            Yhat_idxs = utils.normalize_output(self.net(X_idxs, training=True))
            losses = self.criterion(Y_idxs[:, tf.newaxis],
                                    Yhat_idxs[:, tf.newaxis])  # per example
        dls = inner_tape.jacobian(losses, self.net.trainable_variables)
        # gradient of mean loss w.r.t. parameters
        grad_params = [tf.reduce_mean(dl, axis=0) for dl in dls]
        # gradient of per-example loss w.r.t. parameters
        mean_loss = tf.reduce_mean(losses)
        return mean_loss, grad_params, dls

    def update_recalibrator(self, X_recal, Y_recal):
        """Update the model's recalibrator.

        Also returns the _uncalibrated_ pair (y_pred, y_std) as 1-d arrays.
        """
        Y_pred, Y_std = self.predict(X_recal)
        y_pred, y_std, y_recal = [y.squeeze(1)
                                  for y in [Y_pred, Y_std, Y_recal]]
        self.recalibrator = get_recalibrator(
            y_pred,
            y_std,
            y_recal,
            mode=self.recal_mode,
            **self.recal_kwargs
        )
        return Y_pred, Y_std

    def recalibrate(self, Y_pred, Y_std):
        """Recalibrate predictive uncertainties using the model's
        recalibrator."""
        assert self.recal_mode is not None, (
            "recalibration mode is not set; no recalibrator learned"
        )
        assert (len(Y_pred.shape) == len(Y_std.shape) == 2 and
                Y_pred.shape[1] == Y_std.shape[1] == 1), (
            "inputs to recalibrate() must have shape (n, 1)"
        )

        y_pred, y_std = [y.squeeze(1) for y in [Y_pred, Y_std]]
        y_std = self.recalibrator(y_pred, y_std)
        return np.expand_dims(y_std, 1)

    def _predict_single(self, x, comp_grads=True):
        """A helper for predict() and predict_withGradients() per example."""

        if comp_grads:
            x = tf.Variable(x)
            # variance = mean(influence^2)
            with tf.GradientTape(persistent=True) as outer_tape:
                with tf.GradientTape() as inner_tape:
                    m = utils.ensure_2d(utils.normalize_output(
                        self.net(x, training=False)
                    ))

                grads = inner_tape.gradient(m,
                                            self.net.trainable_variables + [x])
                dmdp, dmdx = grads[:-1], grads[-1]
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

            dmdpdx_higs = outer_tape.gradient(dmdp_higs, x)
            self.dsdx = tf.reduce_mean([
                influence * dmdpdx_hig
                for influence, dmdpdx_hig in zip(influences, dmdpdx_higs)
            ])

            s = tf.math.sqrt(v)
            if s > 0:
                self.dsdx /= s

        # No need for the outer tape in this case
        else:
            with tf.GradientTape() as inner_tape:
                m = utils.ensure_2d(utils.normalize_output(
                    self.net(x, training=False)
                ))
            dmdp = inner_tape.gradient(m, self.net.trainable_variables)

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
            s = tf.math.sqrt(v)

            self.dmdx = None
            self.dsdx = None

        return m, s, self.dmdx, self.dsdx

    def predict(self, X):
        """Get the predicted mean and std at X.

        TODO(yj): batchify this step.
        """
        M, S = [], []
        if self.feature_map is not None:
            X, _ = self.feature_map(X, None)
        else:
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(utils.get_size(X)):
            x = utils.slice_tensors(X, i, i + 1)
            m, s, _, _ = self._predict_single(x, comp_grads=False)
            M.append(m)
            S.append(s.numpy().item())
        M = tf.concat(M, 0).numpy()
        S = np.array(S)[:, np.newaxis]
        return M, S

    def predict_withGradients(self, X):
        """Get the gradients of the predicted mean and variance at X."""
        M, S, dM, dS = [], [], [], []
        if self.feature_map is not None:
            X, _ = self.feature_map(X, None)
        else:
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(utils.get_size(X)):
            x = utils.slice_tensors(X, i, i + 1)
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
