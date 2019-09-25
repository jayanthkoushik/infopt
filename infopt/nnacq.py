"""LCB acquisition function (and optimizer) for neural network model."""

from GPyOpt.acquisitions import AcquisitionLCB
from GPyOpt.experiment_design import RandomDesign
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class NNAcq(AcquisitionLCB):

    """LCB acquisition for neural network model.

    The difference is the use of a gradient descent optimizer.
    """

    def __init__(
        self,
        model,
        space,
        exploration_weight=2,
        optim_cls=Adam,
        optim_kwargs={"lr": 0.01},
        optim_iters=100,
        optim_ckpt_every=1,
        fast_project=None,
        rel_tol=1e-4,
        device=torch.device("cpu"),
        tb_writer=None,
        reinit_optim_start=False,
        lr_decay_step_size=10,
        lr_decay_gamma=0.5,
        x_sampler=None,
    ):
        super().__init__(model, space, exploration_weight=exploration_weight)
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.optim_iters = optim_iters
        self.optim_ckpt_every = optim_ckpt_every
        self.fast_project = fast_project
        self.rel_tol = rel_tol
        self.device = device
        self.tb_writer = tb_writer
        self.reinit_optim_start = reinit_optim_start
        self.x_sampler = x_sampler
        if x_sampler is None:
            self.random_design = RandomDesign(space)
        self.lr_decay_step_size = lr_decay_step_size
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = self
        self.acq_calls = 0
        self.x = None
        self.optim = None
        self.lr_scheduler = None
        self._init_x()
        self.global_iter = 0

    def _init_x(self):
        if self.x_sampler is None:
            x = self.random_design.get_samples(init_points_count=1)
            x_data = torch.from_numpy(x.astype(np.float32)).to(self.device)
        else:
            x_data = self.x_sampler()

        if self.x is None:
            self.x = x_data
            self.x.requires_grad = True
            self.x.grad = torch.zeros_like(self.x, device=self.device)
        else:
            self.x.data.copy_(x_data)
        self.optim = self.optim_cls([self.x], **self.optim_kwargs)
        self.lr_scheduler = StepLR(
            self.optim, self.lr_decay_step_size, self.lr_decay_gamma
        )

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def optimize(self, duplicate_manager=None):
        # pylint: disable=protected-access, unused-argument
        """Override default optimizer."""
        if self.reinit_optim_start:
            self._init_x()

        def _closure():
            nonlocal self
            m, s, dmdx, dsdx = self.model._predict_single(self.x)
            with torch.no_grad():
                _fx = float(m - self.exploration_weight * s)
                self.x.grad.data.copy_(dmdx - self.exploration_weight * dsdx)
            return _fx

        fx = float("inf")
        for iter_ in range(self.optim_iters):
            self.lr_scheduler.step()
            fx, old_fx = self.optim.step(_closure), fx

            if self.tb_writer is not None:
                if (
                    (iter_ + 1) % self.optim_ckpt_every == 0
                ) or iter_ == self.optim_iters - 1:
                    self.tb_writer.add_scalar(
                        "nn_model/acq_optim", fx, self.global_iter
                    )

            # Project point to space
            if self.fast_project is not None:
                self.fast_project(self.x.data)
            else:
                xdata_np = self.x.detach().cpu().numpy()
                xdata_round_np = self.space.round_optimum(xdata_np)
                xdata_round_np = xdata_round_np.astype(np.float32)
                x_round = torch.from_numpy(xdata_round_np).to(self.device)
                self.x.data.copy_(x_round)

            self.global_iter += 1

            # Check tolerance criterion
            if abs(fx - old_fx) < abs(old_fx) * self.rel_tol:
                break

        m, s, _, _ = self.model._predict_single(self.x, comp_grads=False)
        with torch.no_grad():
            fx = float(m - self.exploration_weight * s)
        self.acq_calls += 1
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("nn_model/acq", fx, self.acq_calls)
        return self.x.detach().cpu().numpy(), fx
