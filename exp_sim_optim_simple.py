"""exp_sim_optim_simple.py: a simplified version of exp_sim_optim.py
(synthetic function optimization) *for illustrative purposes*.

Supports both TF2 and PyTorch implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Callable

import tensorflow as tf

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from GPyOpt import Design_space
from GPyOpt.experiment_design import RandomDesign
from GPyOpt.methods import BayesianOptimization
from GPyOpt.core.evaluators import select_evaluator

from exputils.objectives import get_objective
from exputils.optimization import run_optim


"""
Problem Setup
"""


def setup_objective(
        obj_name: str,
        input_dim: int,
        init_points: int = 10,
        n_layers: int = 5,
        n_units: int = 64,
        activation: str = "relu",
        bound: float = 10.0,
        initializer: Callable = torch.nn.init.xavier_uniform_,
):
    """Setup synthetic minimizer objective (e.g., Ackley, Rastrigin, RandomNN).

    Returns (X, Y, domain, problem).
    """
    # RandomNN: initialize & find minimum
    problem_cls = get_objective(obj_name)
    if obj_name.lower() == "randomnn":
        true_layer_sizes = [n_units] * n_layers
        problem = problem_cls(input_dim, true_layer_sizes, activation, bound,
                              initializer)
        if input_dim < 10:
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
    space = Design_space(domain)
    X = RandomDesign(space).get_samples(init_points_count=init_points)
    Y = problem.f(X)
    return X, Y, domain, problem


"""
Optimization helpers
"""


def eval_hook(n, bo, postfix_dict):
    # pylint: disable=unused-argument
    mu_min, sig_min = bo.model.predict(bo.X[np.newaxis, -1])
    mu_min, sig_min = mu_min[-1], sig_min[-1]
    postfix_dict["μ*"] = mu_min.item()
    postfix_dict["σ*"] = sig_min.item()


def report_result(bo):
    mu_min, sig_min = bo.model.predict(bo.x_opt[np.newaxis, :])
    mu_min, sig_min = mu_min.item(), sig_min.item()
    print("predicted minimum value:", mu_min, "+/-", sig_min)
    print("predicted minimizer:", bo.x_opt)
    print("actual value at predicted minimizer:", bo.fx_opt)
    return mu_min, sig_min


def plot_regrets(regrets, data_name,
                 item_names=None, colors=sns.color_palette(), style="seaborn"):
    plt.style.use(style)
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


def get_grid(problem, n_mesh=100):
    assert problem.input_dim == 2
    x1, x2 = [np.linspace(*problem.bounds[i], n_mesh)
              for i in range(problem.input_dim)]
    X = np.stack(np.meshgrid(x1, x2), -1).reshape(-1, problem.input_dim)
    Y = problem.f(X)
    return X, Y


def plot_surface(problem, bound, minimizers, title,
                 n_mesh=100, colors=sns.color_palette(), style="seaborn"):
    plt.style.use(style)
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


"""
Run optimization
"""


def run_sim_optim_gp(
        X,
        Y,
        domain,
        problem,
        max_iter: int = 100,
        acquisition_type: str = "LCB",
        **bo_kwargs,
):
    """Runs standard GP optimization using
    GPyOpt.methods.BayesianOptimization ."""
    myBopt = BayesianOptimization(
        f=problem.f,
        domain=domain,
        model_type="GP",
        acquisition_type=acquisition_type,
        X=X,
        Y=Y,
        **bo_kwargs,
    )

    t0 = time.time()
    myBopt.run_optimization(max_iter=max_iter)
    print("minimum:", myBopt.fx_opt)
    print("minimizer:", myBopt.x_opt)
    print(f"elapsed time: {time.time() - t0:.2f} seconds")
    myBopt.plot_convergence()

    return myBopt


def run_sim_optim_torch(
        X,
        Y,
        domain,
        problem,
        max_iter: int = 100,
        layer_sizes: List[int] = (8, 8, 4),
        ihvp_rank=10,
        ihvp_batch_size=8,
        ihvp_iters_per_point=25,
        ihvp_loss_cls=torch.nn.MSELoss,
        ihvp_optim_cls=torch.optim.Adam,
        ihvp_optim_params={"lr": 0.01},
        nn_optim_cls=torch.optim.Adam,
        nn_optim_params={"lr": 0.01},
        criterion=F.mse_loss,
        update_batch_size=16,
        num_higs=16,
        ihvp_n=16,
        acq_optim_cls=torch.optim.Adam,
        acq_optim_params={"lr": 0.01},
        acq_optim_iters=5000,
        lr_decay_step_size=1000,
        lr_decay_gamma=0.2,
        **kwargs
):
    """Runs NN-INF optimization using torch."""

    # torch specific
    from infopt.ihvp import LowRankIHVP
    from infopt.nnacq import NNAcq
    from infopt.nnmodel import NNModel
    from exputils.models import FCNet

    input_dim = X.shape[1]
    model = FCNet(input_dim, layer_sizes)
    print(model)
    # Use xavier initialization in RandomNN, for fair comparison with tf2
    if problem.__class__.__name__ == "RandomNN":
        for name, param in model.named_parameters():
            if "weight" in name:
                problem.initializer(param)
            if "bias" in name:
                param.data.zero_()

    # IHVP
    ihvp = LowRankIHVP(
        model.parameters(),
        ihvp_rank,
        ihvp_batch_size,
        ihvp_iters_per_point,
        ihvp_loss_cls(),
        ihvp_optim_cls,
        ihvp_optim_params,
    )

    # NN model for the objective
    nn_optim = nn_optim_cls(model.parameters(), **nn_optim_params)
    nn_model = NNModel(
        model,
        ihvp,
        nn_optim,
        criterion=criterion,
        update_batch_size=update_batch_size,
        num_higs=num_higs,
        ihvp_n=ihvp_n,
    )

    # NN acquisition
    space = Design_space(domain)
    acq = NNAcq(
        nn_model,
        space,
        0,  # reset by run_optim
        optim_cls=acq_optim_cls,
        optim_kwargs=acq_optim_params,
        optim_iters=acq_optim_iters,
        lr_decay_step_size=lr_decay_step_size,
        lr_decay_gamma=lr_decay_gamma,
    )

    # Run optimization
    writer = SummaryWriter("logdir_torch")
    import argparse
    optim_args = argparse.Namespace(
        init_points=10,
        optim_iters=max_iter,
        model_update_interval=1,
        exp_multiplier=0.01,
        exp_gamma=0.1,
        use_const_exp_w=None,
        evaluator_cls=select_evaluator("sequential"),
        evaluator_params={},
        tb_writer=writer,
    )
    t0 = time.time()
    result = run_optim(
        problem, space, nn_model, acq, False, optim_args, eval_hook)
    print("Elapsed Time: {:.2f}s".format(time.time() - t0))

    report_result(result["bo"])
    plot_regrets(result["regrets"], problem.name)
    return result


def run_sim_optim_tf2(
        X,
        Y,
        domain,
        problem,
        max_iter: int = 100,
        layer_sizes: List[int] = (8, 8, 4),
        ihvp_rank=10,
        ihvp_batch_size=8,
        ihvp_iters_per_point=25,
        ihvp_criterion=tf.keras.losses.MeanSquaredError(),
        ihvp_optim_cls=tf.keras.optimizers.Adam,
        ihvp_optim_params={"learning_rate": 0.01},
        nn_optim_cls=tf.keras.optimizers.Adam,
        nn_optim_params={"learning_rate": 0.01},
        criterion=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        update_batch_size=16,
        num_higs=16,
        ihvp_n=16,
        acq_optim_cls=tf.keras.optimizers.Adam,
        acq_optim_params={"learning_rate": 0.01},
        acq_optim_iters=5000,
        lr_decay_step_size=1000,
        lr_decay_gamma=0.2,
        **kwargs
):
    """Runs NN-INF optimization using TF2."""

    # TF2 specific
    from infopt.ihvp_tf import LowRankIHVPTF
    from infopt.nnmodel_tf import NNModelTF
    from infopt.nnacq_tf import NNAcqTF
    from exputils.models_tf import make_fcnet

    input_dim = X.shape[1]
    model = make_fcnet(input_dim, layer_sizes)
    model.summary()

    # IHVP
    ihvp = LowRankIHVPTF(
        model.trainable_variables,
        ihvp_rank,
        ihvp_batch_size,
        ihvp_iters_per_point,
        ihvp_criterion,
        ihvp_optim_cls,
        ihvp_optim_params,
    )

    # NN model for the objective
    nn_optim = nn_optim_cls(**nn_optim_params)
    nn_model = NNModelTF(
        model,
        ihvp,
        nn_optim,
        criterion=criterion,
        update_batch_size=update_batch_size,
        num_higs=num_higs,
        ihvp_n=ihvp_n,
    )

    # NN acquisition
    space = Design_space(domain)
    acq = NNAcqTF(
        nn_model,
        space,
        0,
        optim_cls=acq_optim_cls,
        optim_kwargs=acq_optim_params,
        optim_iters=acq_optim_iters,
        lr_decay_step_size=lr_decay_step_size,
        lr_decay_gamma=lr_decay_gamma,
    )

    # Run optimization
    writer = SummaryWriter("logdir_torch")
    import argparse
    optim_args = argparse.Namespace(
        init_points=10,
        optim_iters=max_iter,
        model_update_interval=1,
        exp_multiplier=0.01,
        exp_gamma=0.1,
        use_const_exp_w=None,
        evaluator_cls=select_evaluator("sequential"),
        evaluator_params={},
        tb_writer=writer,
    )
    t0 = time.time()
    result = run_optim(
        problem, space, nn_model, acq, False, optim_args, eval_hook)
    print("Elapsed Time: {:.2f}s".format(time.time() - t0))

    report_result(result["bo"])
    plot_regrets(result["regrets"], problem.name)
    return result
