"""models.py: helpers to create gpyopt models."""

import logging

import GPyOpt.acquisitions
import torch
import torch.nn as nn
from GPyOpt.models import GPModel, GPModel_MCMC

from exputils.optimization import SinglePassAcquisitionOptimizer
from infopt.gpinfacq import GPInfAcq
from infopt.gpinfacq_mcmc import GPInfAcq_MCMC
from infopt.ihvp import LowRankIHVP
from infopt.nnacq import NNAcq
from infopt.nnmodel import NNModel

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def model_nn_inf(base_model, space, args, acq_fast_project=None):
    """Neural network model with influence acquisition."""
    # Returns (model, acq).
    ihvp = LowRankIHVP(
        base_model.parameters(),
        args.ihvp_rank,
        args.ihvp_batch_size,
        args.ihvp_iters_per_point,
        args.ihvp_loss_cls(),
        args.ihvp_optim_cls,
        args.ihvp_optim_params,
        args.ihvp_ckpt_every,
        DEVICE,
        args.tb_writer,
    )

    bo_model_optim = args.bom_optim_cls(
        base_model.parameters(), **args.bom_optim_params
    )
    bo_model = NNModel(
        base_model,
        ihvp,
        bo_model_optim,
        args.bom_loss_cls(),
        args.bom_up_batch_size,
        args.bom_up_iters_per_point,
        args.bom_n_higs,
        args.bom_ihvp_n,
        args.bom_ckpt_every,
        DEVICE,
        args.tb_writer,
    )

    acq = NNAcq(
        bo_model,
        space,
        0,
        args.acq_optim_cls,
        args.acq_optim_params,
        args.acq_optim_iters,
        args.acq_ckpt_every,
        acq_fast_project,
        args.acq_rel_tol,
        DEVICE,
        args.tb_writer,
        args.acq_reinit_optim_start,
        args.acq_optim_lr_decay_step_size,
        args.acq_optim_lr_decay_gamma,
    )

    return bo_model, acq


def model_gp(space, args):
    # TODO: allow setting kernel
    if args.mcmc:
        model = GPModel_MCMC(exact_feval=args.exact_feval, verbose=False)
    else:
        model = GPModel(exact_feval=args.exact_feval, ARD=args.ard, verbose=False)
    logging.debug(f"model: {model}")

    if args.acq_type == "inf":
        acq_cls = GPInfAcq_MCMC if args.mcmc else GPInfAcq
    else:
        acq_cls_name = f"Acquisition{args.acq_type.upper()}"
        if args.mcmc:
            acq_cls_name += "_MCMC"
        acq_cls = getattr(GPyOpt.acquisitions, acq_cls_name)

    acq_optim = SinglePassAcquisitionOptimizer(space)
    acq = acq_cls(model, space, acq_optim)
    logging.info(f"acquisition: {acq}")

    return model, acq


class FCNet(nn.Module):
    def __init__(self, in_dim, layer_sizes, nonlin=torch.relu):
        super().__init__()
        self.in_dim = in_dim
        layer_sizes = [in_dim] + layer_sizes
        self.layers = []
        for i, (ls, ls_n) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append(nn.Linear(ls, ls_n))
            setattr(self, f"fc_{i}", self.layers[-1])
        self.fc_last = nn.Linear(layer_sizes[-1], 1)
        self.scale = nn.Linear(1, 1)
        self.nonlin = nonlin

    def forward(self, x):
        for layer in self.layers:
            x = self.nonlin(layer(x))
        x = torch.tanh(self.fc_last(x))
        x = self.scale(x)
        return x


class NRNet(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, nonlin=torch.relu, init_as_design=True, as_orig=False):
        """Neural network for NeuralUCB & variants.

        init_as_design: If True, initialize weights as per the paper.
        as_orig: If True, add multiplication by `sqrt(m)` at the end as per the paper.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.as_orig = as_orig

        layer_sizes = [in_dim] + [hidden_dim for _ in range(num_layers-1)]
        self.layers = []
        for i, (ls, ls_n) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append(nn.Linear(ls, ls_n, bias=False))
            setattr(self, f"fc_{i}", self.layers[-1])
        self.layers.append(nn.Linear(layer_sizes[-1], 1, bias=False))
        setattr(self, f"fc_{num_layers-1}", self.layers[-1])
        self.nonlin = nonlin

        if init_as_design:
            self._init_weights()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.nonlin(layer(x))
        x = self.layers[-1](x)
        if self.as_orig:
            x *= torch.sqrt(torch.tensor(self.hidden_dim).float())
        return x

    def _init_weights(self):
        """Initialize weights as specified in the paper."""
        def _fill(h, w):
            h_half = h // 2
            w_half = w // 2
            if h > 1:
                out = torch.zeros(h, w)
                W = torch.randn(h_half,w_half).float() * torch.sqrt(torch.tensor(4./w))
                out[:h_half, :w_half] = W
                out[h_half:, w_half:] = W
                return out
            else:
                out = torch.zeros(w)
                W = torch.randn(w_half).float() * torch.sqrt(torch.tensor(2./w))
                out[:w_half] = W
                out[w_half:] = -W
                return torch.unsqueeze(out, 0)

        for layer in self.layers:
            layer.weight.data = _fill(*layer.weight.data.shape)
