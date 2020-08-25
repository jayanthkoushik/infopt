"""gpinfacq.py: influence based acquisition function for GP."""

from collections.abc import Iterator
from contextlib import AbstractContextManager

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase


class GPInfAcq(AcquisitionBase):

    """Lower confidence bound acquisition function, where the ``confidence''
    is estimated using influence functions.

    Arguments are the same as for AcquisitionLCB.
    """

    analytical_gradient_prediction = True

    class InternalModelIterator(AbstractContextManager, Iterator):

        """Iterator over multiple internal models.

        This is not useful here, since there is only one internal model, but is
        used in the MCMC version.
        """

        def __init__(self, bomodel):
            self.bomodel = bomodel
            self._idone = None

        def __enter__(self):
            self._idone = False
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._idone = None

        def __iter__(self):
            if self._idone is None:
                raise RuntimeError("Can't create iterator. Use inside 'with'")
            return self

        def __next__(self):
            if self._idone:
                raise StopIteration
            self._idone = True
            return self.bomodel.model

    def __init__(
        self, model, space, optimizer, cost_withGradients=None, exploration_weight=2
    ):
        super().__init__(model, space, optimizer)
        self.optimizer = optimizer
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print(
                "The set cost function is ignored! Inf. acquisition does not "
                "make sense with cost"
            )

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def _compute_acq(self, x):
        f_acqu, _ = self.__acq_gradients(x, comp_grads=False)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        return self.__acq_gradients(x)

    def __acq_gradients(self, x, comp_grads=True):
        f_acqus = []
        if comp_grads:
            df_acqus = []
        with self.InternalModelIterator(self.model) as int_model_iter:
            for int_model in int_model_iter:
                gp = int_model
                kern = gp.kern
                x_tr = gp.X
                y_tr = gp.Y
                n_tr = len(x_tr)

                x_all = np.vstack([x_tr, x])
                m_all, v_all = gp.predict(x_all, full_cov=True)
                cov = v_all[:n_tr, n_tr:]
                cov = np.clip(cov, 1e-10, np.inf)
                m = m_all[n_tr:]

                r_tr = y_tr - m_all[:n_tr]
                q = n_tr * cov * r_tr
                q_mean = np.mean(q, axis=0, keepdims=True)
                q_0 = q - q_mean
                s_hat = np.std(q, axis=0, keepdims=True, dtype=np.float64).T

                f_acqu = -m + self.exploration_weight * s_hat
                f_acqus.append(f_acqu)

                if comp_grads:
                    K_tr = kern.K(x_tr, x_tr)
                    A = (q_0 * r_tr).T @ (
                        np.eye(n_tr) - K_tr @ gp.posterior.woodbury_inv
                    )
                    Wy = gp.posterior.woodbury_vector[:, 0].T

                    dm = kern.gradients_X(Wy, x, x_tr)
                    ds_hat = kern.gradients_X(A, x, x_tr) / s_hat
                    df_acqu = -dm + self.exploration_weight * ds_hat
                    df_acqus.append(df_acqu)

        mean_f_acqu = np.mean(f_acqus, axis=0)
        if comp_grads:
            mean_df_acqu = np.mean(df_acqus, axis=0)
        else:
            mean_df_acqu = None
        return mean_f_acqu, mean_df_acqu
