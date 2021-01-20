"""nnmodel.py: neural network model for GPyOpt."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from GPyOpt.models.base import BOModel


class NNModel(BOModel):

    """Torch neural network for modeling data."""

    MCMC_sampler = False
    analytical_gradient_prediction = True

    def __init__(
        self,
        net,
        ihvp,
        optim,
        criterion=F.mse_loss,
        update_batch_size=np.inf,
        update_iters_per_point=25,
        num_higs=np.inf,
        ihvp_n=np.inf,
        ckpt_every=1,
        device=torch.device("cpu"),
        tb_writer=None,
    ):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.ihvp = ihvp
        self.higs = []
        self.n = 0
        self.optim = optim
        self.update_batch_size = update_batch_size
        self.update_iters_per_point = update_iters_per_point
        self.num_higs = num_higs
        self.ihvp_n = ihvp_n
        self.ckpt_every = ckpt_every
        self.total_iter = 0
        self.device = device
        self.tb_writer = tb_writer
        self.dsdx = None

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        # Note: X_new and Y_new are ignored
        # pylint: disable=unused-argument
        X = torch.from_numpy(X_all).to(self.device, torch.float32)
        Y = torch.from_numpy(Y_all).to(self.device, torch.float32)
        n_new = len(X) - self.n
        self.n = len(X)

        iters = self.update_iters_per_point * n_new
        batch_size = min(self.n, self.update_batch_size)

        def _closure():
            # Compute predictions and loss
            nonlocal batch_X, batch_Y, self
            batch_Yhat = self.net(batch_X)
            _loss = self.criterion(batch_Yhat, batch_Y)
            self.optim.zero_grad()
            _loss.backward()
            return _loss

        for _ in range(iters):
            batch_idxs = random.sample(range(self.n), batch_size)
            batch_X, batch_Y = X[batch_idxs], Y[batch_idxs]

            loss = self.optim.step(_closure)
            self.total_iter += 1
            if self.tb_writer is not None and self.total_iter % self.ckpt_every == 0:
                self.tb_writer.add_scalar(
                    "nn_model/log_loss", torch.log10(loss).item(), self.total_iter
                )

        ihvp_n = min(self.n, self.ihvp_n)
        idxs = random.sample(range(self.n), ihvp_n)
        X_idxs, Y_idxs = X[idxs], Y[idxs]
        Yhat_idxs = self.net(X_idxs)
        dls = []
        mean_loss = 0
        for y, yhat in zip(Y_idxs, Yhat_idxs):
            l = self.criterion(yhat, y)
            mean_loss += l / ihvp_n

            with torch.no_grad():
                dl = [
                    d.detach()
                    for d in torch.autograd.grad(
                        l, self.net.parameters(), create_graph=True
                    )
                ]
            dls.append(dl)
        self.ihvp.update(mean_loss, dls)

        # Compute H^{-1}grad(L(z)) for selected points
        num_higs = min(len(dls), self.num_higs)
        dls = random.sample(dls, num_higs)
        self.higs = []
        for dl in dls:
            self.higs.append([h.detach() for h in self.ihvp.get_ihvp(dl)])

    def _predict_single(self, x, comp_grads=True):
        x.requires_grad = True
        self.net.zero_grad()
        m = self.net(x)
        dmdp = torch.autograd.grad(m, self.net.parameters(), create_graph=True)
        v = torch.tensor(0.0, device=self.device)
        if comp_grads:
            if self.dsdx is None:
                self.dsdx = torch.zeros_like(x, device=self.device)
            else:
                self.dsdx.zero_()

        for hig in self.higs:
            dmdp_hig = [
                dmdp_i.view(-1) @ hig_i.view(-1) for dmdp_i, hig_i in zip(dmdp, hig)
            ]
            with torch.no_grad():
                inf = sum(dmdp_hig)
                v.add_(inf ** 2 / len(self.higs))

                if comp_grads:
                    dmdpdx_hig = torch.autograd.grad(
                        list(filter(lambda p: p.requires_grad, dmdp_hig)),
                        x,
                        create_graph=True,
                    )[0]
                    self.dsdx.add_(inf * dmdpdx_hig / len(self.higs))

        with torch.no_grad():
            s = torch.sqrt(v)
            if comp_grads:
                if s > 0:
                    self.dsdx.div_(s)
                dmdx = torch.autograd.grad(m, x)[0]
                dsdx = self.dsdx
            else:
                dmdx = None
                dsdx = None
        return m, s, dmdx, dsdx

    def predict(self, X):
        M, S = [], []
        X = torch.from_numpy(X).to(self.device, torch.float32)
        for i in range(len(X)):
            x = X[i : i + 1]
            m, s, _, _ = self._predict_single(x, comp_grads=False)
            M.append(m)
            S.append(s.item())
        M = torch.cat(M).detach().cpu().numpy()
        S = np.array(S)[:, np.newaxis]
        return M, S

    def predict_withGradients(self, X):
        M, S, dM, dS = [], [], [], []
        X = torch.from_numpy(X).to(self.device, torch.float32)
        for i in range(len(X)):
            x = X[i : i + 1]
            m, s, dm, ds = self._predict_single(x, comp_grads=True)
            M.append(m)
            S.append(s.item())
            dM.append(dm)
            dS.append(ds)
        M = torch.cat(M).detach().cpu().numpy()
        S = np.array(S)[:, np.newaxis]
        dM = torch.cat(dM).detach().cpu().numpy()
        dS = torch.cat(dS).detach().cpu().numpy()
        return M, S, dM, dS

    def get_fmin(self):
        raise NotImplementedError

    def get_model_parameters(self):
        """Get parameters to be saved."""
        # pylint: disable=no-self-use
        return []
