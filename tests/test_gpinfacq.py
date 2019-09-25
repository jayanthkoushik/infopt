"""test_gpinfacq.py: test influence based acquisition with GP."""

import os
import sys
import unittest

import GPyOpt
import numpy as np
from scipy.optimize import approx_fprime

from infopt.gpinfacq import GPInfAcq
from infopt.gpinfacq_mcmc import GPInfAcq_MCMC

CHECK_PLACES = 4  # Number of decimal places used for comparing numbers


def _create_sample_model(mcmc=False):
    """Create a sample model by fitting to data from the Ackley function."""
    f = GPyOpt.objective_examples.experimentsNd.ackley(input_dim=10, sd=2)
    X = np.random.uniform(low=-2, high=2, size=(20, 10))

    # f.f has a print statement -_-
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    Y = f.f(X)
    sys.stdout.close()
    sys.stdout = stdout

    bounds = []
    for i in range(10):
        bounds.append({"name": f"var_{i}", "type": "continuous", "domain": (-2, 2)})
    space = GPyOpt.Design_space(bounds)

    if mcmc:
        model = GPyOpt.models.GPModel_MCMC(exact_feval=False, verbose=False)
    else:
        model = GPyOpt.models.GPModel(exact_feval=False, verbose=False, ARD=True)
    model.updateModel(X, Y, None, None)
    return model, space


class TestGPInfAcq(unittest.TestCase):

    """Test cases for GPInfAcq."""

    def setUp(self):
        self.model, self.space = _create_sample_model()

    def test_mean(self):
        """Test just the mean output by setting exploration weight to 0."""
        inf_acq = GPInfAcq(self.model, self.space, None, exploration_weight=0)
        lcb_acq = GPyOpt.acquisitions.AcquisitionLCB(
            self.model, self.space, None, exploration_weight=0
        )

        X = np.random.uniform(low=-2, high=2, size=(100, 10))
        inf_acqs, inf_dacqs = inf_acq.acquisition_function_withGradients(X)
        lcb_acqs, lcb_dacqs = lcb_acq.acquisition_function_withGradients(X)

        for mi, ml in zip(inf_acqs, lcb_acqs):
            mi, ml = mi.item(), ml.item()
            self.assertAlmostEqual(
                mi, ml, places=CHECK_PLACES, msg="inf, lcb mean not equal"
            )

        for di, dl in zip(inf_dacqs, lcb_dacqs):
            for di_j, dl_j in zip(di, dl):
                di_j, dl_j = di_j.item(), dl_j.item()
                self.assertAlmostEqual(
                    di_j,
                    dl_j,
                    places=CHECK_PLACES,
                    msg="inf, lcb mean gradients not equal",
                )

    def _test_grad(self, w, noassert=False):
        """Test acquisition gradient by comparing with numerical approx."""
        X = np.random.uniform(low=-2, high=2, size=(100, 10))
        inf_acq = GPInfAcq(self.model, self.space, None, exploration_weight=w)
        errs = []

        for x in X:
            _, grad_calc = inf_acq.acquisition_function_withGradients(x[np.newaxis, :])
            grad_calc = grad_calc[0]
            grad_est = approx_fprime(
                x, lambda x: inf_acq.acquisition_function(x[np.newaxis, :]).item(), 1e-6
            )
            errs.append(np.max(np.abs(grad_calc - grad_est)))

        if noassert:
            print(
                f"error: mean {np.mean(errs):.3g}, "
                f"min {min(errs):.3g}, max {max(errs):.3g}"
            )
        else:
            self.assertAlmostEqual(
                np.mean(errs),
                0,
                places=CHECK_PLACES,
                msg="analytical and numerical gradients do not match "
                f"(exploration weight {w}) "
                f"min err {min(errs):.3g}, max_err {max(errs):.3g}",
            )

    def test_grad_w0(self):
        """Test grad with w=0."""
        self._test_grad(w=0)

    def test_grad_w1(self):
        """Test grad with w=1."""
        self._test_grad(w=1, noassert=True)


class TestGPInfAcq_MCMC(unittest.TestCase):

    """Test cases for GPInfAcq_MCMC.

    Since the MCMC version just uses a different internal model iterator,
    it is sufficient to test that.
    """

    def setUp(self):
        self.model, _ = _create_sample_model(mcmc=True)

    def test_model_iterator_correctness(self):
        """Test GPInfAcq_MCMC.InternalModelIterator."""
        X = np.random.uniform(low=-2, high=2, size=(100, 10))
        means_true, _ = self.model.predict(X)
        means_comp = []
        init_params = self.model.model.param_array.copy()
        with GPInfAcq_MCMC.InternalModelIterator(self.model) as int_model_iter:
            for int_model in int_model_iter:
                m, _ = int_model.predict(X)
                means_comp.append(m)
        final_params = self.model.model.param_array.copy()

        for pi, pf in zip(init_params.flatten(), final_params.flatten()):
            self.assertAlmostEqual(
                pi,
                pf,
                places=CHECK_PLACES,
                msg="model parameters not same after " "iteration",
            )

        for mean_true, mean_comp in zip(means_true, means_comp):
            for mt, mc in zip(mean_true[:, 0], mean_comp[:, 0]):
                self.assertAlmostEqual(
                    mt,
                    mc,
                    places=CHECK_PLACES,
                    msg="prediction computed with internal "
                    "model iterator does not match truth",
                )

    def test_ensure_context_req(self):
        """Ensure that the model iterator cannot be used without context."""
        with self.assertRaises(
            Exception, msg="Exception not raised for use " "without context"
        ):
            next(GPInfAcq_MCMC.InternalModelIterator(self.model))
