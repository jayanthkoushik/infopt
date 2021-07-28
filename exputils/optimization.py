"""optimization.py: gpyopt wrappers for optimization."""

import logging
import time
from functools import partial

import numpy as np
import torch
from GPyOpt.core.task import SingleObjective
from GPyOpt.experiment_design import initial_design, RandomDesign
from GPyOpt.methods import ModularBayesianOptimization
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from GPyOpt.optimization.optimizer import apply_optimizer, choose_optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class SinglePassAcquisitionOptimizer(AcquisitionOptimizer):

    """Acquisition optimizer which does a single pass from a random start."""

    def __init__(self, space, optimizer="lbfgs", **kwargs):
        # pylint: disable=super-init-not-called, unused-argument
        self.space = space
        self.optimizer_name = optimizer
        self.context_manager = ContextManager(space)
        self.random_design = RandomDesign(space)

    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None):
        # pylint: disable=attribute-defined-outside-init
        self.f = f
        self.df = df
        self.f_df = f_df
        self.optimizer = choose_optimizer(
            self.optimizer_name, self.context_manager.noncontext_bounds
        )
        anchor_point = self.random_design.get_samples(init_points_count=1)
        x_min, fx_min = apply_optimizer(
            self.optimizer,
            anchor_point,
            f=f,
            df=None,
            f_df=f_df,
            duplicate_manager=duplicate_manager,
            context_manager=self.context_manager,
            space=self.space,
        )
        return x_min, fx_min


def run_optim(fun, space, model, acq, normalize_Y, args, eval_hook=None):
    """Optimize `fun` over `space` using `model` and `acq`."""
    np.random.seed()

    obj = SingleObjective(fun.f)
    eval_ = args.evaluator_cls(acq, **args.evaluator_params)
    init_sample = initial_design("random", space, args.init_points)
    bo = ModularBayesianOptimization(
        model,
        space,
        obj,
        acq,
        eval_,
        init_sample,
        normalize_Y=normalize_Y,
        model_update_interval=args.model_update_interval,
    )

    fmin_hat = np.inf
    inst_regrets = []
    regrets = []
    iter_times = []
    mu_min, sig_min = [], []
    mu_at_min, sig_at_min = [], []

    def new_evaluate_objective(self):
        nonlocal fmin_hat, inst_regrets, regrets, iter_times
        nonlocal timer, postfix_dict, pbar, args
        nonlocal mu_min, sig_min, mu_at_min, sig_at_min
        self.evaluate_objective_orig()
        pbar.update(1)

        iter_times.append(time.time() - timer)
        timer = time.time()
        postfix_dict["t_i"] = iter_times[-1]

        if hasattr(acq, "exploration_weight") and args.use_const_exp_w is None:
            _t = pbar.n + 1
            acq.exploration_weight = (
                args.exp_multiplier * np.sqrt(_t) * (np.log(_t / args.exp_gamma) ** 2)
            )
            postfix_dict["exp_w"] = acq.exploration_weight

        if hasattr(fun, "f_noiseless"):
            _fhat = fun.f_noiseless(self.X[np.newaxis, -1, :]).item()
            fmin_hat = min(fmin_hat, _fhat)
            if fun.fmin is not None:
                inst_regrets.append(_fhat - fun.fmin)
                regrets.append(fmin_hat - fun.fmin)
                postfix_dict["r_i"] = inst_regrets[-1]
                postfix_dict["R_i"] = regrets[-1]
            postfix_dict["f*hat"] = fmin_hat
            postfix_dict["y_i"] = self.Y[-1].item()
            postfix_dict["fhat_i"] = _fhat

        if eval_hook is not None:
            eval_hook(pbar.n, bo, postfix_dict)
            if "μ*" in postfix_dict:
                mu_min.append(postfix_dict["μ*"])
                sig_min.append(postfix_dict["σ*"])
            if "μ(x*)" in postfix_dict:
                mu_at_min.append(postfix_dict["μ(x*)"])
                sig_at_min.append(postfix_dict["σ(x*)"])

        if args.tb_writer is not None:
            if fun.fmin is not None:
                args.tb_writer.add_scalar("regret", regrets[-1], pbar.n)
            else:
                args.tb_writer.add_scalar("fstar_hat", fmin_hat, pbar.n)

        pbar.set_postfix(**postfix_dict)

    bo.evaluate_objective_orig = bo.evaluate_objective
    bo.evaluate_objective = partial(new_evaluate_objective, self=bo)

    postfix_dict = dict()
    if hasattr(acq, "exploration_weight"):
        if args.use_const_exp_w is not None:
            acq.exploration_weight = args.use_const_exp_w
            postfix_dict["exp_w"] = args.use_const_exp_w
            logging.info(
                f"using constant exploration weight {args.use_const_exp_w:.3g}"
            )
        else:
            init_exp_w = args.exp_multiplier * (np.log(1 / args.exp_gamma) ** 2)
            acq.exploration_weight = init_exp_w
            postfix_dict["exp_w"] = init_exp_w
            logging.info(f"set initial exploration weight to {init_exp_w:.3g}")
    elif hasattr(acq, "jitter"):
        acq.jitter = 0
        logging.info("set jitter to 0")

    if fun.fmin is not None:
        logging.info(f"f*={fun.fmin:.3g}")

    timer = time.time()
    with trange(args.optim_iters, desc="Optimizing", leave=True) as pbar:
        bo.run_optimization(args.optim_iters, eps=-1)

    result = {"iter_times": iter_times, "X": bo.X, "y": bo.Y, "bo": bo,
              "runtime": time.time() - timer}
    if inst_regrets:
        result["inst_regrets"] = inst_regrets
        result["regrets"] = regrets
    if mu_min:
        result["μ*"] = mu_min
        result["σ*"] = sig_min
    if mu_at_min:
        result["μ(x*)"] = mu_at_min
        result["σ(x*)"] = sig_at_min
    return result


def optimize_nn_offline(model, X, y, space, args):
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    )
    data_loader = DataLoader(
        dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    device = next(model.parameters()).device

    tr_optimizer = args.train_optim_cls(model.parameters(), **args.train_optim_params)
    tr_lr_sched = StepLR(
        tr_optimizer, args.train_lr_decay_step_size, args.train_lr_decay_gamma
    )
    tr_batch_iter = iter(data_loader)
    tr_loss = args.train_loss_cls()

    with trange(args.train_iters, desc="Training network") as pbar:
        for _ in pbar:
            try:
                X_batch, y_batch = next(tr_batch_iter)
            except StopIteration:
                tr_batch_iter = iter(data_loader)
                X_batch, y_batch = next(tr_batch_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            yhat_batch = model(X_batch)
            loss = tr_loss(yhat_batch, y_batch)

            tr_optimizer.zero_grad()
            loss.backward()
            tr_lr_sched.step()
            tr_optimizer.step()
            pbar.set_postfix(loss=float(loss))

    x_optim_data = RandomDesign(space).get_samples(init_points_count=1)
    x_optim = torch.from_numpy(x_optim_data.astype(np.float32)).to(device)
    x_optim.requires_grad = True

    opt_optimizer = args.opt_optim_cls([x_optim], **args.opt_optim_params)
    opt_lr_sched = StepLR(
        opt_optimizer, args.opt_lr_decay_step_size, args.opt_lr_decay_gamma
    )
    x_optim_best = None
    fx_optim_best = float("inf")

    with trange(args.opt_iters, desc="Optimizing output") as pbar:
        for _ in pbar:
            fx_optim = model(x_optim)

            opt_optimizer.zero_grad()
            fx_optim.backward()
            opt_lr_sched.step()
            opt_optimizer.step()

            x_optim_data = x_optim.detach().cpu().numpy()
            x_optim_data_round = space.round_optimum(x_optim_data).astype(np.float32)
            x_optim.data.copy_(torch.from_numpy(x_optim_data_round).to(device))

            fx_optim = float(fx_optim)
            pbar.set_postfix(fx_optim=fx_optim)
            if fx_optim < fx_optim_best:
                fx_optim_best = fx_optim
                x_optim_best = x_optim_data_round[0]

    return x_optim_best, fx_optim_best
