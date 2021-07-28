"""exp_sim_optim_simple.py: a simplified version of exp_sim_optim.py
(synthetic function optimization) *for illustrative purposes*.

Supports both TF2 and PyTorch implementations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
from scipy.stats import describe
import seaborn as sns
import time
from typing import Callable

import tensorflow as tf

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import GPyOpt
from GPyOpt import Design_space
from GPyOpt.experiment_design import RandomDesign
from GPyOpt.methods import BayesianOptimization
from GPyOpt.core.evaluators import select_evaluator

from exputils.objectives import get_objective
from exputils.optimization import run_optim

# torch specific
from infopt.ihvp import LowRankIHVP
from infopt.nnacq import NNAcq
from infopt.nnmodel import NNModel
from exputils.models import FCNet

# TF2 specific
from infopt.ihvp_tf import LowRankIHVPTF
from infopt.nnmodel_tf import NNModelTF
from infopt.nnmodel_mcd_tf import NNModelMCDTF
from infopt.nnacq_tf import NNAcqTF
from exputils.models_tf import make_fcnet

"""
Problem Setup
"""


def setup_objective(
        obj_name: str,
        input_dim: int,
        n_layers: int = 5,
        n_units: int = 64,
        activation: str = "relu",
        bound: float = 10.0,
        initializer: Callable = torch.nn.init.xavier_uniform_,
):
    """Setup synthetic minimizer objective (e.g., Ackley, Rastrigin, RandomNN).

    Returns (problem, domain, X, Y).
    """
    # RandomNN: initialize & find minimum
    problem_cls = get_objective(obj_name)
    if obj_name.lower() == "randomnn":
        true_layer_sizes = [n_units] * n_layers
        problem = problem_cls(input_dim, true_layer_sizes, activation, bound,
                              initializer)
        if input_dim < 4:
            problem.find_fmin()  # grid search
        else:
            problem.find_fmin_optim()  # L-BFGS
    else:
        problem = problem_cls(input_dim)

    domain = [
        {'name': f'var_{i+1}',
         'type': 'continuous',
         'domain': problem.bounds[i]}
        for i in range(input_dim)
    ]
    return problem, domain


"""
Run optimization
"""


def run_sim_optim_gp(
        problem,
        domain,
        max_iter=100,
        acquisition_type="LCB",
        init_points=1,
        optimizer_restarts=1,
        verbosity=True,
        plot_filename=None,
        save_filename=None,
        report_filename=None,
        evaluations_filename=None,
        **bo_kwargs,
):
    """Runs standard GP optimization via GPyOpt.methods.BayesianOptimization"""
    myBopt = BayesianOptimization(
        f=problem.f,
        domain=domain,
        model_type="GP",
        acquisition_type=acquisition_type,
        optimizer_restarts=optimizer_restarts,
        initial_design_numdata=init_points,
        **bo_kwargs,
    )

    t0 = time.time()
    myBopt.run_optimization(
        max_iter=max_iter,
        verbosity=verbosity,
        report_file=report_filename,
        evaluations_file=evaluations_filename,
    )
    print("minimum:", myBopt.fx_opt)
    print("minimizer:", myBopt.x_opt)
    mu_at_min, sig_at_min = myBopt.model.predict(np.array([problem.min]))
    print("μ(x*):", mu_at_min.item())
    print("σ(x*):", sig_at_min.item())
    print(f"elapsed time: {time.time() - t0:.2f} seconds")
    if plot_filename is not None:
        myBopt.plot_convergence(filename=plot_filename)
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        with open(save_filename, "wb") as f:
            pickle.dump(myBopt, f)

    return myBopt


def run_sim_optim_torch(
        problem,
        domain,
        layer_sizes=[8, 8, 4],
        dropout=0.0,
        ihvp_rank=10,
        ihvp_batch_size=8,
        ihvp_iters_per_point=25,
        ihvp_loss_cls=torch.nn.MSELoss,
        ihvp_optim_cls=torch.optim.Adam,
        ihvp_optim_params={"lr": 0.01},
        nn_optim_cls=torch.optim.Adam,
        nn_optim_params={"lr": 0.02, "weight_decay": 1e-5},
        criterion=F.mse_loss,
        acq_optim_cls=torch.optim.Adam,
        acq_optim_params={"lr": 0.05},
        **kwargs
):
    """Runs NN-INF optimization using torch."""
    input_dim = problem.input_dim
    base_model = FCNet(input_dim, layer_sizes, dropout=dropout)
    print(base_model)
    # Use xavier initialization in RandomNN, for fair comparison with tf2
    if problem.__class__.__name__ == "RandomNN":
        for name, param in base_model.named_parameters():
            if "weight" in name:
                problem.initializer(param)
            if "bias" in name:
                param.data.zero_()

    # IHVP
    ihvp = LowRankIHVP(
        base_model.parameters(),
        ihvp_rank,
        ihvp_batch_size,
        ihvp_iters_per_point,
        ihvp_loss_cls(),
        ihvp_optim_cls,
        ihvp_optim_params,
    )

    return run_sim_optim_nn(
        problem,
        domain,
        base_model,
        NNModel,
        NNAcq,
        ihvp=ihvp,
        nn_optim_cls=nn_optim_cls,
        nn_optim_params=nn_optim_params,
        criterion=criterion,
        acq_optim_cls=acq_optim_cls,
        acq_optim_params=acq_optim_params,
        tb_writer=SummaryWriter("logdir_sim_optim/nn_inf_torch"),
        **kwargs
    )


def run_sim_optim_tf2(
        problem,
        domain,
        layer_sizes=(8, 8, 4),
        dropout=0.0,
        ihvp_rank=10,
        ihvp_batch_size=8,
        ihvp_iters_per_point=25,
        ihvp_criterion=tf.keras.losses.MeanSquaredError(),
        ihvp_optim_cls=tf.keras.optimizers.Adam,
        ihvp_optim_params={"learning_rate": 0.01},
        **kwargs
):
    """Runs NN-INF optimization using TF2."""
    input_dim = problem.input_dim
    base_model = make_fcnet(input_dim, layer_sizes, dropout=dropout)
    base_model.summary()

    # IHVP
    ihvp = LowRankIHVPTF(
        base_model.trainable_variables,
        ihvp_rank,
        ihvp_batch_size,
        ihvp_iters_per_point,
        ihvp_criterion,
        ihvp_optim_cls,
        ihvp_optim_params,
    )

    return run_sim_optim_nn(
        problem,
        domain,
        base_model,
        NNModelTF,
        NNAcqTF,
        ihvp=ihvp,
        tb_writer=SummaryWriter("logdir_sim_optim/nn_inf_tf2"),
        **kwargs
    )


def run_sim_optim_mcd_tf2(
        problem,
        domain,
        layer_sizes=(8, 8, 4),
        dropout=0.1,
        n_dropout_samples=100,
        lengthscale=1e-2,
        tau=0.25,
        **kwargs
):
    """Runs NN-MCD optimization using TF2."""
    input_dim = problem.input_dim
    base_model = make_fcnet(input_dim, layer_sizes, dropout=dropout)
    base_model.summary()

    return run_sim_optim_nn(
        problem,
        domain,
        base_model,
        NNModelMCDTF,
        NNAcqTF,
        tb_writer=SummaryWriter("logdir_sim_optim/nn_mcd_tf2"),
        **kwargs,
        # MCD specific
        dropout=dropout,
        n_dropout_samples=n_dropout_samples,
        lengthscale=lengthscale,
        tau=tau,
    )


def run_sim_optim_nn(
        problem,
        domain,
        base_model,
        nn_model_cls,
        nn_acq_cls,
        ihvp=None,
        max_iter=100,
        normalize_Y=False,
        nn_optim_cls=tf.keras.optimizers.Adam,
        nn_optim_params={"learning_rate": 0.02},
        criterion=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        update_batch_size=16,
        update_iters_per_point=25,
        ckpt_every=10,
        num_higs=8,
        ihvp_n=15,
        acq_optim_cls=tf.keras.optimizers.Adam,
        acq_optim_params={"learning_rate": 0.05},
        acq_optim_iters=5000,
        acq_optim_lr_decay_step_size=1000,
        acq_optim_lr_decay_gamma=0.2,
        acq_rel_tol=1e-3,
        acq_reinit_optim_start=True,
        init_points=1,
        model_update_interval=1,
        exp_multiplier=0.1,
        exp_gamma=0.1,
        use_const_exp_w=None,
        evaluator_cls=select_evaluator("sequential"),
        evaluator_params={},
        tb_writer=SummaryWriter("logdir_sim_optim"),
        plot_filename=None,
        save_filename=None,
        **nn_model_kwargs,
):
    """Generic routine for NN-based optimization (NN-INF, NN-MCD, etc.).

    Default arguments are for TF2.
    For torch, change the optimization and criterion classes.
    """

    # NN model for the objective
    if issubclass(nn_optim_cls, torch.optim.Optimizer):
        nn_optim = nn_optim_cls(base_model.parameters(), **nn_optim_params)
    else:
        nn_optim = nn_optim_cls(**nn_optim_params)
    if ihvp is not None:
        nn_model = nn_model_cls(
            base_model,
            ihvp,
            nn_optim,
            criterion=criterion,
            update_batch_size=update_batch_size,
            update_iters_per_point=update_iters_per_point,
            ckpt_every=ckpt_every,
            num_higs=num_higs,
            ihvp_n=ihvp_n,
            **nn_model_kwargs,
        )
    else:
        nn_model = nn_model_cls(
            base_model,
            nn_optim,
            criterion=criterion,
            update_batch_size=update_batch_size,
            update_iters_per_point=update_iters_per_point,
            ckpt_every=ckpt_every,
            **nn_model_kwargs,
        )

    # NN acquisition
    space = Design_space(domain)
    acq = nn_acq_cls(
        nn_model,
        space,
        0,
        optim_cls=acq_optim_cls,
        optim_kwargs=acq_optim_params,
        optim_iters=acq_optim_iters,
        lr_decay_step_size=acq_optim_lr_decay_step_size,
        lr_decay_gamma=acq_optim_lr_decay_gamma,
        rel_tol=acq_rel_tol,
        reinit_optim_start=acq_reinit_optim_start,
    )

    # Run optimization
    optim_args = argparse.Namespace(
        init_points=init_points,
        optim_iters=max_iter,
        model_update_interval=model_update_interval,
        exp_multiplier=exp_multiplier,
        exp_gamma=exp_gamma,
        use_const_exp_w=use_const_exp_w,
        evaluator_cls=evaluator_cls,
        evaluator_params=evaluator_params,
        tb_writer=tb_writer,
    )
    t0 = time.time()
    result = run_optim(
        problem, space, nn_model, acq, normalize_Y, optim_args, eval_hook)
    t1 = time.time()
    print("Elapsed Time: {:.2f}s".format(t1 - t0))
    result["runtime"] = t1 - t0
    report_result(result["bo"])

    if plot_filename is not None:
        plot_regrets(result["regrets"], problem.name,
                     plot_filename=plot_filename)
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        del result["bo"]
        with open(save_filename, "wb") as f:
            pickle.dump(result, f)
    return result


"""
Optimization helpers
"""


def eval_hook(n, bo, postfix_dict):
    # pylint: disable=unused-argument
    mu_min, sig_min = bo.model.predict(bo.X[np.newaxis, -1])  # prediction at the last acquisition (lowest LCB)
    mu_min, sig_min = mu_min[-1], sig_min[-1]
    postfix_dict["μ*"] = mu_min.item()
    postfix_dict["σ*"] = sig_min.item()
    postfix_dict["α*"] = (
        bo.acquisition.acquisition_function(bo.X[np.newaxis, -1]).item()
    )
    mu_at_min, sig_at_min = bo.model.predict(
        np.zeros((1, bo.X.shape[1])))  # TODO: any fmin
    postfix_dict["μ(x*)"] = mu_at_min.item()
    postfix_dict["σ(x*)"] = sig_at_min.item()
    exp_w = bo.acquisition.exploration_weight
    postfix_dict["α(x*)"] = (
            postfix_dict["μ(x*)"] - exp_w * postfix_dict["σ(x*)"]
    )

def report_result(bo):
    mu_min, sig_min = bo.model.predict(bo.x_opt[np.newaxis, :])
    mu_min, sig_min = mu_min.item(), sig_min.item()
    print("predicted minimum value:", mu_min, "+/-", sig_min)
    print("predicted minimizer:", bo.x_opt)
    print("actual value at predicted minimizer:", bo.fx_opt)
    return mu_min, sig_min


def plot_regrets(regrets, data_name,
                 item_names=None, colors=sns.color_palette(),
                 plot_style="seaborn-whitegrid",
                 plot_filename=None):
    plt.style.use(plot_style)
    plt.figure(figsize=(8, 5))
    if type(regrets[0]) == list or isinstance(regrets[0], np.ndarray):
        assert item_names is not None
        T = len(regrets[0])
        for rs, item_name, color in zip(regrets, item_names, colors):
            plt.plot(np.arange(T), rs, color=color, label=item_name)
        plt.legend(loc="lower left", bbox_to_anchor=(1.15, 0.0))
    else:
        T = len(regrets)
        plt.plot(np.arange(T), regrets)
    plt.title(f"Regrets for {data_name}")
    plt.xlabel("Iterations")
    plt.ylabel(r"$f(\hat{x}) - f^*$")
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)


def get_grid(problem, n_mesh=100):
    assert problem.input_dim == 2
    x1, x2 = [np.linspace(*problem.bounds[i], n_mesh)
              for i in range(problem.input_dim)]
    X = np.stack(np.meshgrid(x1, x2), -1).reshape(-1, problem.input_dim)
    Y = problem.f(X)
    return X, Y


def plot_surface(problem, bound, minimizers, title,
                 n_mesh=100, colors=sns.color_palette(),
                 plot_style="seaborn",
                 plot_filename=None):
    plt.style.use(plot_style)
    plt.figure(figsize=(8, 5))
    X, Y = get_grid(problem, n_mesh=n_mesh)
    ax = plt.scatter(X[:, 0], X[:, 1], c=Y)
    for i, (x, fx, name) in enumerate(minimizers):
        print(f"{name:16}: min {fx:.5f} at {x}")
        plt.scatter(*x, color=colors[i], label=f"{name}, {fx:.4f}")
        # plt.text(*x, s=f"{name}, {fx:.4f}", color=colors[i])
    plt.title(title)
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    plt.colorbar(ax)
    plt.legend(bbox_to_anchor=(1.6, 1.0))
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)


def plot_acq(bo, filename=None, label_x=None, label_y=None):
    from GPyOpt.plotting.plots_bo import plot_acquisition
    plot_acquisition(bo.acquisition.space.get_bounds(),
                     bo.X.shape[1],
                     bo.model,
                     bo.X,
                     bo.Y,
                     bo.acquisition.acquisition_function,
                     bo.suggest_next_locations(),
                     filename,
                     label_x,
                     label_y)


def plot_pointwise_acq(bo,
                       normalize_Y=False,
                       result_dict=None,
                       name="",
                       colors=sns.color_palette(),
                       plot_style="seaborn-whitegrid",
                       plot_filename=None):
    """Plot acquisition for the initial & acquired points only."""
    mus, sigs = bo.model.predict(bo.X)
    if isinstance(bo, GPyOpt.methods.ModularBayesianOptimization):
        init_points = bo.X.shape[0] - bo.num_acquisitions
    else:
        init_points = bo.initial_design_numdata
    times = np.arange(-init_points, bo.X.shape[0] - init_points)
    ys = GPyOpt.util.general.normalize(bo.Y) if normalize_Y else bo.Y
    mus, sigs, ys = [a.flatten() for a in [mus, sigs, ys]]

    plt.style.use(plot_style)
    plt.figure(figsize=(10, 5))
    plt.plot(times, ys,
             "o", alpha=0.7, color="gray", label=r"$y_t$")
    plt.plot(times, mus, '.-', alpha=0.7)
    plt.fill_between(times, mus - sigs, mus + sigs,
                     alpha=0.5, label=r"$\hat\mu_T(x_t) \pm \hat\sigma_T(x_t)$")
    if result_dict is not None:
        mu_mins, sig_mins = np.array(result_dict['μ*']), np.array(result_dict['σ*'])
        plt.fill_between(times[init_points:], mu_mins - sig_mins, mu_mins + sig_mins,
                         alpha=0.5, label=r"$\hat\mu_t(x_t) \pm \hat\sigma_t(x_t)$")
        try:
            mu_at_mins, sig_at_mins = np.array(result_dict['μ(x*)']), np.array(result_dict['σ(x*)'])
            plt.fill_between(times[init_points:], mu_at_mins - sig_at_mins, mu_at_mins + sig_at_mins,
                             alpha=0.5, label=r"$\hat\mu_t(x^*) \pm \hat\sigma_t(x^*)$")
        except KeyError:
            print("μ(x*) not found, skipping")
    plt.axvline(0, c="gray", alpha=0.7)
    plt.axvline(times[mus.argmin()], c=colors[-1], alpha=0.7, label="$\hat{y}_{min}$")
    plt.title("Predicted mean and acquisition over initial/acquired points" + f": {name}")
    plt.xlabel(r"$t$")
    plt.ylabel("value")
    plt.legend(loc=4, bbox_to_anchor=(1.2, 0))
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)


def plot_sigs(bo, name="", plot_style="seaborn-whitegrid", plot_filename=None):
    plt.style.use(plot_style)
    plt.figure(figsize=(8, 5))

    # acquired points
    mus, sigs = bo.model.predict(bo.X)
    n, input_dim = bo.X.shape
    print("acquired:", describe(sigs))

    # random points
    mus_r, sigs_r = bo.model.predict(RandomDesign(bo.space).get_samples(n))
    print("random:", describe(sigs_r))

    # points near zero (optimum)
    mus_z, sigs_z = bo.model.predict(np.random.uniform(-1, 1,
                                                       size=(n, input_dim)))
    print("near minimum:", describe(sigs_z))

    plt.hist(sigs, bins=12, alpha=0.7, label="acquired")
    plt.hist(sigs_r, bins=12, alpha=0.7, label="random")
    plt.hist(sigs_z, bins=12, alpha=0.7, label="near minimum")
    plt.xlabel(r"$\sigma(x)$")
    plt.ylabel("count")
    plt.legend()

    plt.title(f"Predictive uncertainties for {name}")
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)
