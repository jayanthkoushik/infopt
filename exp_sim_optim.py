"""exp_sim_optim.py: test various optimizers on standard functions."""

import logging
import os
import pickle
import sys
from contextlib import redirect_stdout
from functools import wraps
from glob import glob
from math import pi, sqrt

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    ClassType,
    CommaSeparatedInts,
    KeyValuePairsType,
    LazyHelpFormatter,
    OutputFileType,
)
from shinyutils.matwrap import MatWrap as mw

import GPyOpt
import numpy as np
import pandas as pd
from clearml import Task as ClearMLTask
from GPyOpt import Design_space
from GPyOpt.experiment_design import RandomDesign
from GPyOpt.objective_examples.experiments1d import forrester
from GPyOpt.objective_examples.experiments2d import (
    beale,
    dropwave,
    eggholder,
    powers,
    sixhumpcamel,
)
from GPyOpt.objective_examples.experimentsNd import ackley, alpine1, alpine2
from scipy.linalg import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from torch.nn.modules.loss import _Loss, MSELoss
from torch.optim import Adam, Optimizer

from exputils.models import DEVICE, FCNet, model_gp, model_nn_inf, model_nr, NRNet
from exputils.optimization import optimize_nn_offline, run_optim
from exputils.parsing import (
    base_arg_parser,
    make_gp_parser,
    make_nn_nr_parser,
    make_optim_parser,
    make_plot_parser,
    TensorboardWriterType,
)
from exputils.plotting import plot_performance, plot_timing
from exputils.objectives import Ackley


def main():
    """Entry point."""
    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    run_parser = sub_parsers.add_parser("run", formatter_class=LazyHelpFormatter)
    run_parser.set_defaults(func=run)
    run_parser.add_argument("--save-file", type=OutputFileType(), required=True)
    run_parser.add_argument(
        "--fname", type=str, required=True, metavar="function", choices=FS
    )
    run_parser.add_argument("--fdim", type=int, required=True)
    make_optim_parser(run_parser)
    run_parser.add_argument(
        "--tb-dir", type=TensorboardWriterType(), dest="tb_writer", default=None
    )

    noise_parser = run_parser.add_argument_group("output noise")
    noise_var_group = noise_parser.add_mutually_exclusive_group(required=False)
    noise_var_group.add_argument("--gaussian-noise-with-scale", type=float)
    noise_var_group.add_argument("--gprbf-noise-with-scale", type=float)
    noise_var_group.add_argument("--xnorm-gaussian-noise-with-scale", type=float)

    run_sub_parsers = run_parser.add_subparsers(dest="mname")
    run_sub_parsers.required = True

    gp_parser = run_sub_parsers.add_parser("gp", formatter_class=LazyHelpFormatter)
    make_gp_parser(gp_parser)

    for _mname in ["nn", "nr"]:
        mname_parser = run_sub_parsers.add_parser(
            _mname, formatter_class=LazyHelpFormatter
        )
        mname_parser.add_argument(
            "--layer-sizes",
            type=CommaSeparatedInts(),
            required=True,
            metavar="int,int[,...]",
        )
        mname_parser.add_argument("--dropout", type=float, default=0.0)
        make_nn_nr_parser(mname_parser, _mname)

    run_offnn_parser = sub_parsers.add_parser(
        "run-offline-nn", formatter_class=LazyHelpFormatter
    )
    run_offnn_parser.set_defaults(func=run_offline_nn)
    run_offnn_parser.add_argument("--res-dirs", type=str, nargs="+", required=True)
    run_offnn_parser.add_argument("--total-jobs", type=int, default=None)

    offnn_train_parser = run_offnn_parser.add_argument_group(
        "offline training parameters"
    )
    offnn_train_parser.add_argument("--batch-size", type=int, default=64)
    offnn_train_parser.add_argument("--train-iters", type=int, default=5000)
    offnn_train_parser.add_argument(
        "--train-loss-cls", type=ClassType(_Loss), metavar="loss", default=MSELoss
    )
    offnn_train_parser.add_argument(
        "--train-optim-cls",
        type=ClassType(Optimizer),
        metavar="optimizer",
        default=Adam,
    )
    offnn_train_parser.add_argument(
        "--train-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.01),
    )
    offnn_train_parser.add_argument(
        "--train-lr-decay-step-size", type=int, default=2000
    )
    offnn_train_parser.add_argument("--train-lr-decay-gamma", type=float, default=0.1)

    offnn_opt_parser = run_offnn_parser.add_argument_group(
        "offline optimization parameters"
    )
    offnn_opt_parser.add_argument("--opt-iters", type=int, default=10000)
    offnn_opt_parser.add_argument(
        "--opt-optim-cls", type=ClassType(Optimizer), metavar="optimizer", default=Adam
    )
    offnn_opt_parser.add_argument(
        "--opt-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.05),
    )
    offnn_opt_parser.add_argument("--opt-lr-decay-step-size", type=int, default=2000)
    offnn_opt_parser.add_argument("--opt-lr-decay-gamma", type=float, default=0.2)

    plotr_parser = sub_parsers.add_parser(
        "plot-regrets", formatter_class=LazyHelpFormatter
    )
    plotr_parser.set_defaults(func=plot_performance)
    make_plot_parser(plotr_parser)

    plott_parser = sub_parsers.add_parser(
        "plot-timing", formatter_class=LazyHelpFormatter
    )
    plott_parser.set_defaults(func=plot_timing)
    make_plot_parser(plott_parser)
    plott_parser.add_argument("--model-update-interval", type=int, default=1)

    ploto_parser = sub_parsers.add_parser(
        "plot-offline-nn", formatter_class=LazyHelpFormatter
    )
    ploto_parser.set_defaults(func=plot_offline_nn)
    make_plot_parser(ploto_parser)

    args = base_arg_parser.parse_args()
    if args.clearml_task:
        ClearMLTask.init(project_name="infopt_sim_optim", task_name=args.clearml_task)
        logging.info(
            f"clearml logging enabled under 'infopt_sim_optim/{args.clearml_task}'"
        )
    if os.path.exists(args.save_file.name) and os.path.getsize(args.save_file.name) > 0:
        logging.info("save file %s exists, skipping", args.save_file.name)
    else:
        args.func(args)


def run(args):
    funf = FS[args.fname]
    try:
        fun = funf(args.fdim)
    except RuntimeError as e:
        logging.critical(e)
        sys.exit(1)
    logging.info(f"target function: {args.fname}-{args.fdim}d")

    # figure out function range
    if not hasattr(fun, "range"):
        if args.fname == "ackley":
            with redirect_stdout(open(os.devnull, "w")):
                _urange = fun.f(np.array([[_b[1] for _b in fun.bounds]])).item()
        elif args.fname == "alpine1":
            _urange = 11 * args.fdim
        elif args.fname == "alpine2":
            _urange = sqrt(10) ** args.fdim
        elif args.fname == "beale":
            _urange = fun.f(np.array([[-1, -1]])).item()
        elif args.fname == "dropwave":
            _urange = 0
        elif args.fname == "eggholder":
            _urange = 1071
        elif args.fname == "forrester":
            _urange = 16
        elif args.fname == "powers":
            _urange = 2
        elif args.fname == "sixhumpcamel":
            _urange = fun.f(np.array([[2, 1]])).item()
        else:
            raise AssertionError
        fun.range = [fun.fmin, _urange]
    logging.info(f"function range: {fun.range}")

    # Add noise
    if args.gaussian_noise_with_scale is not None:
        assert args.gaussian_noise_with_scale > 0
        _scale = args.gaussian_noise_with_scale * (fun.range[1] - fun.range[0]) / 2
        noise_fun = GaussianNoise(_scale)
    elif args.gprbf_noise_with_scale is not None:
        assert args.gprbf_noise_with_scale > 0
        _scale = args.gprbf_noise_with_scale * (fun.range[1] - fun.range[0]) / 2
        noise_fun = GPNoise(RBF(), _scale)
    elif args.xnorm_gaussian_noise_with_scale is not None:
        assert args.xnorm_gaussian_noise_with_scale > 0
        _scale = (
            args.xnorm_gaussian_noise_with_scale * (fun.range[1] - fun.range[0]) / 2
        )
        noise_fun = XScaleNoise(_scale)
    else:
        noise_fun = None
    logging.info(f"noise: {noise_fun}")

    fun.f_noiseless = fun.f
    if noise_fun is not None:
        # Decorate fun.f to add noise to its output
        def decorate_with_noise(_f, _noisef):
            @wraps(_f)
            def _wrapper(_x):
                _fx = _f(_x)
                _nx = _noisef(_x)
                assert _fx.shape == _nx.shape
                return _fx + _nx

            return _wrapper

        fun.f = decorate_with_noise(fun.f, noise_fun)

    bounds = []
    for i, bound in enumerate(fun.bounds):
        bounds.append({"name": f"x_{i}", "type": "continuous", "domain": bound})
    space = Design_space(bounds)

    if args.mname == "gp":
        _exact_feval = noise_fun is None
        logging.info(
            f"{'exact' if _exact_feval else 'noisy'} evaluation setting for gp model"
        )
        model, acq = model_gp(space, args, _exact_feval)
        normalize_Y = True
    else:
        b0 = fun.bounds[0]
        if all((b[0] == b0[0] and b[1] == b0[1]) for b in fun.bounds[1:]):
            logging.info(f"using fast project with bounds {b0}")
            acq_fast_project = lambda x: x.clamp_(*b0)
        else:
            logging.info("using default projection")
            acq_fast_project = None

        if args.mname == "nr":
            if any(l != args.layer_sizes[0] for l in args.layer_sizes):
                logging.critical("NRNet needs all hidden layer sizes to be the same")
                sys.exit(1)
            base_model = NRNet(
                len(args.layer_sizes),
                args.fdim,
                args.layer_sizes[0],
                as_orig=args.use_original_nr,
                has_bias=args.nr_has_bias,
            ).to(DEVICE)
            model, acq = model_nr(base_model, space, args, acq_fast_project)
        else:
            base_model = FCNet(args.fdim, args.layer_sizes, dropout=args.dropout).to(DEVICE)
            model, acq = model_nn_inf(base_model, space, args, acq_fast_project)

        logging.debug(base_model)
        normalize_Y = False

    mu_mins, sig_mins, acq_mins, y_mins, mse, test_mse = [], [], [], [], [], []
    mu_at_mins, sig_at_mins, acq_at_mins, exp_w = [], [], [], []
    X_test = RandomDesign(space).get_samples(init_points_count=200)
    with redirect_stdout(open(os.devnull, "w")):
        Y_test = fun.f(X_test)

    def eval_hook(n, bo, postfix_dict):
        # pylint: disable=unused-argument
        nonlocal mu_mins, sig_mins, acq_mins, y_mins, mse, test_mse
        nonlocal mu_at_mins, sig_at_mins, acq_at_mins, exp_w
        nonlocal fun
        nonlocal X_test, Y_test

        m, s = bo.Y.mean(), bo.Y.std()

        def _unnormalize(_mus, _sigs):
            if s > 0:
                _mus *= s
                _sigs *= s
            _mus += m
            return _mus, _sigs

        # TODO: handle mcmc case
        # prediction at the last acquisition (lowest LCB by model)
        mus, sigs = bo.model.predict(bo.X)
        if bo.normalize_Y:
            mus, sigs = _unnormalize(mus, sigs)
        mu_min, sig_min = mus[-1], sigs[-1]
        _exp_w = bo.acquisition.exploration_weight
        acq_min = -mu_min + _exp_w * sig_min
        mu_mins.append(mu_min.item())
        sig_mins.append(sig_min.item())
        acq_mins.append(acq_min.item())
        exp_w.append(_exp_w)
        postfix_dict["μ*"] = mu_mins[-1]
        postfix_dict["σ*"] = sig_mins[-1]
        postfix_dict["α*"] = acq_mins[-1]

        # Y_min & MSE (for t-1)
        ys = bo.Y[:-1]
        y_mins.append(ys.min())
        postfix_dict["y_min"] = y_mins[-1]
        mse.append(np.mean((mus[:-1] - ys) ** 2))
        postfix_dict["mse"] = mse[-1]
        # Test MSE
        mus_test, sigs_test = bo.model.predict(X_test)
        if bo.normalize_Y:
            mus_test, sigs_test = _unnormalize(mus_test, sigs_test)
        test_mse.append(np.mean((mus_test - Y_test) ** 2))
        postfix_dict["test_mse"] = test_mse[-1]

        # prediction at actual minimum
        if fun.fmin is not None:
            mu_at_min, sig_at_min = bo.model.predict(np.array([fun.min]))
            if bo.normalize_Y:
                mu_at_min, sig_at_min = _unnormalize(mu_at_min, sig_at_min)
            acq_at_min = -mu_min + _exp_w * sig_min
            mu_at_mins.append(mu_at_min.item())
            sig_at_mins.append(sig_at_min.item())
            acq_at_mins.append(acq_at_min.item())
            postfix_dict["μ(x*)"] = mu_at_mins[-1]
            postfix_dict["σ(x*)"] = sig_at_mins[-1]
            postfix_dict["α(x*)"] = acq_at_mins[-1]

    if args.fname == "ackley":
        # ackley has a print statement inside -_-
        with redirect_stdout(open(os.devnull, "w")):
            result = run_optim(fun, space, model, acq, normalize_Y, args, eval_hook)
    else:
        result = run_optim(fun, space, model, acq, normalize_Y, args, eval_hook)

    save_file = args.save_file
    del args.save_file
    del args.func
    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer
    del result["bo"]
    result["fmin"] = fun.fmin
    result["args"] = vars(args)
    result.update({
        "mu_mins": mu_mins,
        "sig_mins": sig_mins,
        "acq_mins": acq_mins,
        "exp_w": exp_w,
        "mse": mse,
        "test_mse": test_mse,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_mins": y_mins,
        "mu_at_mins": mu_at_mins,
        "sig_at_mins": sig_at_mins,
        "acq_at_mins": acq_at_mins,
    })
    pickle.dump(result, save_file.buffer)
    save_file.close()


def run_offline_nn(args):
    jobs_done = 0
    for res_dir in args.res_dirs:
        for res_file in glob(os.path.join(res_dir, "**", "*.pkl"), recursive=True):
            with open(res_file, "rb") as f:
                res = pickle.load(f)

            mname = res["args"]["mname"]
            if mname != "nn":
                continue
            logging.info(f"generating offline results for {res_file}")

            fun = FS[res["args"]["fname"]](res["args"]["fdim"])
            bounds = [
                {"name": f"x_{i}", "type": "continuous", "domain": bound}
                for i, bound in enumerate(fun.bounds)
            ]
            space = Design_space(bounds)
            net = FCNet(res["args"]["fdim"], res["args"]["layer_sizes"]).to(DEVICE)

            xopt, fxopt = optimize_nn_offline(net, res["X"], res["y"], space, args)
            res["offline_xopt"] = xopt
            res["offline_fxopt"] = fxopt

            with open(res_file, "wb") as f:
                pickle.dump(res, f)

            jobs_done += 1
            if args.total_jobs is not None:
                logging.info(f"[{jobs_done}/{args.total_jobs}] jobs completed")


def plot_offline_nn(args):
    if args.skip_pats:
        raise RuntimeError("skip-patterns not allowed")

    df_rows = []
    for res_file in glob(os.path.join(args.res_dir, "**", "*.pkl"), recursive=True):
        with open(res_file, "rb") as f:
            res = pickle.load(f)

        mname = res["args"]["mname"]
        if mname != "nn":
            continue

        init_points = res["args"]["init_points"]
        iters = res["args"]["optim_iters"]
        offline_fxopt = res["offline_fxopt"]

        for t in range(iters):
            df_row = {
                "id": res_file,
                "t": t,
                "r": (res["y"][: init_points + t + 1].min().item() - offline_fxopt),
            }
            df_rows.append(df_row)

    data_df = pd.DataFrame(df_rows)

    mw.configure(args.context, args.style, args.font)
    fig = mw.plt().figure()
    ax = fig.add_subplot(111)
    if args.log_scale:
        ax.set_yscale("log", nonposy="mask")
    mw.sns().lineplot(
        x="t",
        y="r",
        data=data_df,
        palette=mw.palette()[:1],
        markers=False,
        units="id",
        estimator=None,
        legend=False,
        ax=ax,
    )
    ax.yaxis.tick_right()
    ax.set_xlabel("$t$")
    ax.set_ylabel("Relative regret")
    fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)


class rastrigin:

    """Rastrigin function.

    This is a scaled version:

        f(x) = A + (1/n)*sum_{i=1}^n [x_i^2 - A*cos(2*π*x_i)]
    """

    def __init__(self, n, A=10.0, bound=5.12):
        self.bounds = [(-bound, bound)] * n
        self.min = [0.0] * n
        self.fmin = 0.0
        self.input_dim = n
        self.A = A
        self.range = [0, 2 * self.A + bound * bound]

    def f(self, X):
        X = GPyOpt.util.general.reshape(X, self.input_dim)
        return self.A + np.mean(
            X ** 2 - self.A * np.cos(2 * pi * X), axis=1, keepdims=True
        )


class rosenbrock:

    """Rosenbrock function.

    Multidimensional extension (with scale γ) defined as:

        f(x) = γ/(n-1)*sum_{i=1}^{n-1} [A*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    """

    def __init__(self, n, A=100.0, bound=5, scale=1e-3):
        self.bounds = [(-bound, bound)] * n
        self.min = [1.0] * n
        self.fmin = 0
        self.input_dim = n
        self.A = A
        self.range = [0, 1 + 2 * bound + (4 * A + 1) * bound * bound]
        self.scale = scale

    def f(self, X):
        X = GPyOpt.util.general.reshape(X, self.input_dim)
        return self.scale * np.mean(
            self.A * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2,
            axis=1,
            keepdims=True,
        )


class branin:

    """2 dimensional Branin function.

    Definition (x and y refer to the two coordinates):

        f(x) = a(y - b*x^2 + c*x - r)^2 + s*(1 - t)*cos(x) + s
    """

    def __init__(
        self,
        n,
        a=1.0,
        b=(5.1 / (4 * pi * pi)),
        c=(5 / pi),
        r=6,
        s=10,
        t=(1 / (8 * pi)),
        scale=0.1,
    ):
        if n != 2:
            raise RuntimeError(f"Branin function only defined for n=2 (given n={n})")
        self.input_dim = n
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t
        self.scale = scale
        self.bounds = [(-5, 10), (0, 15)]
        self.range = [0, scale * (a * ((15 + c * 10 - r) ** 2) + s * (2 - t))]
        self.fmin = scale * 0.397887

    def f(self, X):
        X = GPyOpt.util.general.reshape(X, self.input_dim)
        x, y = X[:, 0], X[:, 1]
        return (
            self.scale
            * (
                self.a * ((y - self.b * x * x + self.c * x - self.r) ** 2)
                + self.s * (1 - self.t) * np.cos(x)
                + self.s
            )[:, np.newaxis]
        )


def fix_init_input_dim(cls_, fixed_input_dim):
    """Update the __init__ for cls_ to force constant input_dim."""
    cls_._old_init = cls_.__init__

    def _new_init(self, input_dim):
        # pylint: disable=unused-argument
        if input_dim != fixed_input_dim:
            raise RuntimeError(
                f"{cls_.__name__} function only defined for "
                f"n={fixed_input_dim} "
                f"(given n={input_dim})"
            )
        self._old_init()

    cls_.__init__ = _new_init


fix_init_input_dim(beale, 2)
fix_init_input_dim(dropwave, 2)
fix_init_input_dim(eggholder, 2)
fix_init_input_dim(forrester, 1)
fix_init_input_dim(powers, 2)
fix_init_input_dim(sixhumpcamel, 2)


FS = {
    "ackley": Ackley,
    "alpine1": alpine1,
    "alpine2": alpine2,
    "beale": beale,
    "branin": branin,
    "dropwave": dropwave,
    "eggholder": eggholder,
    "forrester": forrester,
    "powers": powers,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "sixhumpcamel": sixhumpcamel,
}


class GaussianNoise:

    """Gaussian 0-mean noise."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return np.random.normal(scale=self.scale, size=(len(x), 1))

    def __repr__(self):
        return f"Gaussian(mean=0, std={self.scale:.3g})"


class GPNoise:

    """Gaussian process noise distribution."""

    def __init__(self, kernel, scale):
        self.gp = GaussianProcessRegressor(kernel)
        self.kernel = kernel
        self.scale = scale

    def __call__(self, x):
        noise_cov = np.diag((self.scale * self.gp.sample_y(x)[:, 0]) ** 2)
        noise_mean = np.zeros(len(x))
        noise = np.random.multivariate_normal(noise_mean, noise_cov)
        return noise[:, np.newaxis]

    def __repr__(self):
        return f"GP(mean=0, kernel={self.kernel}, scale={self.scale:.3g})"


class XScaleNoise:

    """Heteroskedastic noise that scales with norm of input."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        xscale = self.scale * norm(x, ord=2, axis=-1, keepdims=True)
        return np.random.normal(scale=xscale)

    def __repr__(self):
        return f"||x|| * Gaussian(mean=0, std={self.scale:.3g})"


if __name__ == "__main__":
    main()
