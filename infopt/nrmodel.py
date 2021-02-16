"""nnmodel.py: neural network model for GPyOpt."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from GPyOpt.models.base import BOModel


class NRModel(BOModel):
    """Torch neural network for modeling data, based on NeuralUCB.
    """
    MCMC_sampler = False
    analytical_gradient_prediction = True

    def __init__(
        self,
        net,
        lda,
        optim,
        criterion=F.mse_loss,
        update_batch_size=np.inf,
        update_iters_per_point=25,
        ckpt_every=1,
        device=torch.device("cpu"),
        tb_writer=None,
    ):
        """
        lda: Hyperparameter lambda
        """
        super().__init__()
        self.net = net
        self.lda = lda
        self.criterion = criterion
        self.n = 0
        self.optim = optim
        self.update_batch_size = update_batch_size
        self.update_iters_per_point = update_iters_per_point
        self.ckpt_every = ckpt_every
        self.total_iter = 0
        self.device = device
        self.tb_writer = tb_writer
        self.dsdx = None

        # Calculate value for `m`
        # NOTE: Extended definition: choose the minimum of all hidden feature dimensions
        self.m = np.min([layer.out_features for layer in self.net.layers if layer.out_features > 1])

        # Initialize Z-inverse matrix
        self.p = np.sum([np.prod(param.shape) for param in net.parameters()
                         if param.requires_grad])
        self.Z_inv = 1./lda * torch.eye(self.p)

    @staticmethod
    def _flatten_grads(grads):
        """Flattens parameter gradients from multiple layers into a single vector."""
        return torch.cat([grad.view(-1) for grad in grads])
    
    def _update_Z_inv(self, out_grads):
        """Use Sherman-Morrison formula to expedite computation of inverse,
        since Z is updated by adding a rank 1 matrix which is the outer product of grad.
        Idea borrowed from https://github.com/sauxpa/neural_exploration.
        """
        flat_out_grad = self.__class__._flatten_grads(out_grads)
        ivp = torch.matmul(self.Z_inv, flat_out_grad)
        denom = 1 + torch.matmul(flat_out_grad, ivp)
        self.Z_inv -= 1./denom * torch.outer(ivp, ivp)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        NOTE: `Y_new` is ignored; `X_new` is optional but highly recommended.
        NOTE: In accordance with Algorithm 1, update of Z matrix comes
              before model training, EVEN for the case of multiple new points
              (just like in their synthetic experiment 7.1 setup).
        NOTE: In the paper the model is trained using GD; however, we do (S)GD here
              using constant `update_iters_per_point` and `update_batch_size` values.
              By default, `update_batch_size=np.inf` which is equivalent to GD.
        """
        # pylint: disable=unused-argument
        X = torch.from_numpy(X_all).to(self.device, torch.float32)
        Y = torch.from_numpy(Y_all).to(self.device, torch.float32)
        n_new = len(X) - self.n
        self.n = len(X)

        # Update Z-inverse matrix using each of the new points
        if X_new is None:
            idxs = random.sample(range(self.n), min(self.n, 10))
            X_new = X[idxs]
        else:
            X_new = torch.from_numpy(X_new).to(self.device, torch.float32)
        Yhat_new = self.net(X_new)
        for yhat in Yhat_new:
            single_out_grads = [
                d.detach()
                for d in torch.autograd.grad(
                    yhat, self.net.parameters(), create_graph=True
                )
            ]
            self._update_Z_inv(single_out_grads)

        if self.tb_writer is not None and self.total_iter % self.ckpt_every == 0:
            self.tb_writer.add_scalar(
                "nr_model/Z_inv_elemwise_avg", torch.mean(self.Z_inv).item(), self.total_iter
            )

        # Update model using all the collected data points
        iters = self.update_iters_per_point * n_new
        batch_size = min(self.n, self.update_batch_size)

        def _closure():
            # Compute loss with L2 penalty w.r.t. initial weights
            nonlocal batch_X, batch_Y, self
            batch_Yhat = self.net(batch_X)
            _loss = self.criterion(batch_Yhat, batch_Y) * 0.5
            regularizer = 0
            for p_new, p_old in zip(self.net.parameters(), self.net.params_0):
                regularizer += torch.sum((p_new - p_old)**2)
            _loss += self.m * self.lda * regularizer * 0.5
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
                    "nr_model/log_loss", torch.log10(loss).item(), self.total_iter
                )

    def _uncertainty_mat(self):
        return self.Z_inv

    def _predict_single(self, x, comp_grads=True):
        x.requires_grad = True
        self.net.zero_grad()
        m = self.net(x)
        flat_dmdp = self.__class__._flatten_grads(
                   torch.autograd.grad(m, self.net.parameters(), create_graph=True)
               )
        v = torch.matmul(flat_dmdp, torch.matmul(self._uncertainty_mat(), flat_dmdp))

        with torch.no_grad():
            s = torch.sqrt(v)
            if comp_grads:
                self.dsdx = torch.autograd.grad(v, x, retain_graph=True)[0]
                if s > 0:
                    self.dsdx.div_(2*s)  # NOTE: `self.dsdx` was actually dvdx up to this point
                dmdx = torch.autograd.grad(m, x)[0]
                dsdx = self.dsdx
            else:
                dmdx = None
                dsdx = None
        return m.detach(), s, dmdx, dsdx

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
            ret = self._predict_single(x, comp_grads=True)
            m, s, dm, ds = ret
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


class NRIFModel(NRModel):
    def __init__(
        self,
        net,
        lda,
        optim,
        criterion=F.mse_loss,
        update_batch_size=np.inf,
        update_iters_per_point=25,
        ckpt_every=1,
        device=torch.device("cpu"),
        tb_writer=None,
    ):
        super().__init__(
            net, lda, optim, criterion,
            update_batch_size, update_iters_per_point,
            ckpt_every, device, tb_writer
        )

        # Initialize L matrix
        self.L = torch.eye(self.p) * lda

    def _update_L(self, action, reward):
        """Recursively updates the L matrix given (optimal action, observed reward) pair.
        Also keep a record of the number of updates (i.e. number of training points)
        for computing confidence interval.

        action: Action selected as optimal
        reward: Observed reward for the optimal action
        returns: None
        """
        self.net.zero_grad()
        out = self.net(action)
        loss = self.criterion(out, torch.unsqueeze(reward, 0))
        loss.backward()
        loss_grads = [param.grad for param in self.net.parameters() if param.requires_grad]
        flat_loss_grad = self.__class__._flatten_grads(loss_grads).detach()
        self.L += torch.outer(flat_loss_grad, flat_loss_grad)

    def _uncertainty_mat(self):
        return torch.matmul(torch.matmul(self.Z_inv, self.L), self.Z_inv)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        NOTE: `X_new`, `Y_new` are optional but highly recommended.
        NOTE: In accordance with Algorithm 1, update of Z matrix comes
              before model training, EVEN for the case of multiple new points
              (just like in their synthetic experiment 7.1 setup).
        NOTE: In the paper the model is trained using GD; however, we do (S)GD here
              using constant `update_iters_per_point` and `update_batch_size` values.
              By default, `update_batch_size=np.inf` which is equivalent to GD.
        """
        # pylint: disable=unused-argument
        X = torch.from_numpy(X_all).to(self.device, torch.float32)
        Y = torch.from_numpy(Y_all).to(self.device, torch.float32)
        n_new = len(X) - self.n
        self.n = len(X)

        # Update Z-inverse & L matrices using each of the new points
        if X_new is None:
            idxs = random.sample(range(self.n), min(self.n, 10))
            X_new = X[idxs]
            Y_new = Y[idxs]
        else:
            X_new = torch.from_numpy(X_new).to(self.device, torch.float32)
        Yhat_new = self.net(X_new)
        for x, yhat, y in zip(X_new, Yhat_new, Y_new):
            single_out_grads = [
                d.detach()
                for d in torch.autograd.grad(
                    yhat, self.net.parameters(), create_graph=True
                )
            ]
            self._update_Z_inv(single_out_grads)
            self._update_L(x, y)

        if self.tb_writer is not None and self.total_iter % self.ckpt_every == 0:
            self.tb_writer.add_scalar(
                "nr_model/Z_inv_elemwise_avg", torch.mean(self.Z_inv).item(), self.total_iter
            )

        # Update model using all the collected data points
        iters = self.update_iters_per_point * n_new
        batch_size = min(self.n, self.update_batch_size)

        def _closure():
            # Compute loss with L2 penalty w.r.t. initial weights
            nonlocal batch_X, batch_Y, self
            batch_Yhat = self.net(batch_X)
            _loss = self.criterion(batch_Yhat, batch_Y) * 0.5
            regularizer = 0
            for p_new, p_old in zip(self.net.parameters(), self.net.params_0):
                regularizer += torch.sum((p_new - p_old)**2)
            _loss += self.m * self.lda * regularizer * 0.5
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
                    "nr_model/log_loss", torch.log10(loss).item(), self.total_iter
                )
