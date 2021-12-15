"""nnmodel_mcd_tf.py: a TensorFlow neural network model with MC Dropout
 for GPyOpt."""

import logging
import time
import numpy as np
import tensorflow as tf

from GPyOpt.models.base import BOModel

import uncertainty_toolbox as uct

from infopt import utils
from infopt.recalibration import get_recalibrator


class NNModelMCDTF(BOModel):
    """TensorFlow neural network for modeling the objective function.

    The variance of the lower confidence bound (LCB) is estimated using
    Monte Carlo Dropout (Gal & Ghahramani, 2016).

    Expected to be used in conjunction with infopt.nnacq_tf.NNAcqTF .
    """

    def __init__(
        self,
        net,    # tf.keras.models.Model
        optim,  # tf.keras.optimizers.Optimizer
        criterion=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        update_batch_size=np.inf,
        update_iters_per_point=25,
        update_max_iters=np.inf,
        update_upsample_new=None,
        early_stopping=False,
        dropout=0.05,  # defined in net?
        n_dropout_samples=100,
        lengthscale=1e-2,
        tau=1.0,
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
        self.optim = optim
        self.criterion = criterion
        assert (hasattr(self.criterion, "reduction") and
                (self.criterion.reduction == tf.keras.losses.Reduction.NONE)), (
            f"criterion must have reduction=='none', "
            f"but got {self.criterion.reduction}"
        )
        self.early_stopping = early_stopping
        # MCD parameters
        self.dropout = dropout
        self.n_dropout_samples = n_dropout_samples
        self.lengthscale = lengthscale
        self.tau = tau

        self.n = 0
        self.update_batch_size = update_batch_size
        self.update_iters_per_point = update_iters_per_point
        self.update_max_iters = update_max_iters
        self.update_upsample_new = update_upsample_new
        self.recal_mode = recal_mode
        self.recal_setsize = recal_setsize
        self.recal_kwargs = recal_kwargs
        self.recalibrator = None
        self.ckpt_every = ckpt_every
        self.total_iter = 0
        self.device = device
        self.tb_writer = tb_writer

        # idx -> features for categorical selection
        self.feature_map = kwargs.get("feature_map", None)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """Train the NN using (X_all, Y_all) and get new IHVP estimates."""
        # Note: X_new and Y_new are ignored
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

        # Train NN using current data points
        iters = min(self.update_iters_per_point * n_new, self.update_max_iters)
        batch_size = min(self.n, self.update_batch_size)

        # Change from torch: shuffle all data once before iterating through all
        shuffled_batches = data.shuffle(128).repeat().batch(batch_size)
        # L2 regularizer determined by MC dropout
        reg = (self.lengthscale ** 2 * (1 - self.dropout) /
               (2. * self.n * self.tau))

        old_loss = 1e-8
        t0 = time.time()
        for i, (batch_X, batch_Y) in shuffled_batches.enumerate():
            # Standard TF2 train step + L2 regularization
            with tf.GradientTape() as tape:
                batch_Yhat = utils.normalize_output(
                    self.net(batch_X, training=True))
                loss = tf.reduce_mean(
                    self.criterion(batch_Y[:, tf.newaxis],
                                   batch_Yhat[:, tf.newaxis])
                )
                loss += reg * tf.reduce_sum([
                    tf.reduce_sum(v ** 2) for v in self.net.trainable_variables
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

        # update recalibrator & compute calibration error
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

        # x is tf.Tensor([1, *input_dims]) or tf.RaggedTensor([1, *input_dims])
        assert x.shape[0] == 1, (
            "_predict_single only accepts input tensors with first dimension 1"
            f", got {x.shape[0]}"
        )

        data = tf.data.Dataset.from_tensor_slices(x)
        T = self.n_dropout_samples
        batch_size = min(T, self.update_batch_size)

        preds, grads = [], []
        # Batchify forwards over T dropout samples
        for batch in data.repeat(T).batch(batch_size):
            with tf.GradientTape() as tape:
                if comp_grads:
                    tape.watch(batch)
                yt_hat = utils.ensure_2d(utils.normalize_output(
                    self.net(batch, training=True)
                ))
            preds.append(yt_hat)
            if comp_grads:
                dytdx = tape.gradient(yt_hat, batch)
                grads.append(dytdx)

        # Predictive mean & variance using dropout samples
        preds = tf.concat(preds, axis=0)  # T x 1
        m = tf.reduce_mean(preds, axis=0, keepdims=True)  # 1 x 1
        v = (tf.math.reduce_variance(preds, axis=0, keepdims=True)
             + (1. / self.tau))  # 1 x 1
        s = tf.math.sqrt(v)  # 1 x 1

        # Their gradients
        if comp_grads:
            grads = tf.concat(grads, axis=0)  # T x 1 x input_dim -> 1 x input_dim
            dmdx = tf.reduce_mean(grads, axis=0, keepdims=True)  # 1 x input_dim
            # dsdx = dvdx / s, by chain rule (constants 2 cancelled out)
            dvdx = tf.reduce_mean((preds - m) * grads, axis=0)  # 1 x input_dim
            dsdx = dvdx / s  # 1 x input_dim
        else:
            dmdx = None
            dsdx = None

        return m, s, dmdx, dsdx

    def predict(self, X):
        """Get the predicted mean and std at X using MC dropout.

        T is the number of dropout samples obtained per input.
        """
        # M, S = [], []
        # if self.feature_map is not None:
        #     X, _ = self.feature_map(X, None)
        # else:
        #     X = tf.convert_to_tensor(X, dtype=tf.float32)
        # for i in range(utils.get_size(X)):
        #     x = utils.slice_tensors(X, i, i + 1)
        #     m, s, _, _ = self._predict_single(x, comp_grads=False)
        #     M.append(m)
        #     S.append(s.numpy().item())
        # M = tf.concat(M, 0).numpy()
        # S = np.array(S)[:, np.newaxis]
        # return M, S

        # Faster
        if self.feature_map is not None:
            X, _ = self.feature_map(X, None)
        else:
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        data = tf.data.Dataset.from_tensor_slices(X)
        batch_size = min(utils.get_size(X), self.update_batch_size)

        # (len(X), T)
        T = self.n_dropout_samples
        predictions = tf.concat([
            utils.normalize_output(self.net(batch_X, training=True))  # dropout!
            for batch_X in data.repeat(T).batch(batch_size)
        ], axis=0)
        predictions = tf.transpose(tf.reshape(predictions,
                                              (T, utils.get_size(X))))

        M = tf.reduce_mean(predictions, axis=1, keepdims=True).numpy()
        var = 1. / self.tau + \
            tf.math.reduce_variance(predictions, axis=1, keepdims=True).numpy()
        S = np.sqrt(var)
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
