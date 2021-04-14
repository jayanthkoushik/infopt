"""ihvp_tf.py: inverse Hessian-vector product for TensorFlow neural nets.

HVP Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py
"""

from abc import ABC, abstractmethod
from tqdm import tqdm

import tensorflow as tf


class BaseIHVPTF(ABC):

    """Base class for computing Hessian-vector products in TensorFlow.

    Similar construction as the torch equivalent (`infopt.ihvp.BaseIHVP`),
    but importantly requires an active outer (second-order) gradient tape
    that is used to calculate `self.out`.
    """

    def __init__(self):
        self.out = None
        self.params = None
        self.grad_params = None
        self.outer_tape = None

    @abstractmethod
    def get_ihvp(self, v):
        """Compute [H_params(self.out)]^(-1)v."""
        raise NotImplementedError

    # pylint: disable=unused-argument
    def update(self, out, grad_params, outer_tape):
        """Update the output, its gradients, and the outer gradient tape
        with respect to which ihvp is computed."""
        self.out = out
        self.grad_params = grad_params
        self.outer_tape = outer_tape

    def close(self):
        """Closes the outer gradient tape to free up memory.

        To be called after computing all ihvp's in the current training loop."""
        self.grad_params = None
        if self.outer_tape is None:
            return
        self.outer_tape._tape = None


class IterativeIHVPTF(BaseIHVPTF):

    """Combination of Taylor expansion with Perlmutter's method for
    comuting inverse Hessian-vector product approximation.

    It is important that the Hessian is positive definite, and has spectral
    norm not greater than 1.

    A TensorFlow counterpart to `infopt.ihvp.IterativeIHVP`.
    """

    def __init__(self, params, iters, reg=0.0, scale=1.0):
        """
        Arguments:
            params: list of parameter variables. The Hessian is computed wrt
                these.
            iters: iterations for approximation.
            reg: regularization added to the Hessian.
            scale: number by which the Hessian is divided.
        """
        super().__init__()
        self.params = list(params)
        self.n = len(self.params)
        self.iters = iters
        self.reg = reg
        self.scale = scale

    def compute_hvp(self, v):
        """Computes the HVP knowing gradients and their outer tape.

        v has the same shape as self.params and self.grad_params.
        """
        assert isinstance(v, list)
        assert len(v) == self.n
        assert self.grad_params is not None and self.outer_tape is not None
        # back-over-back hvp using the output_gradients option
        hvp = [tf.debugging.check_numerics(h_i, "hvp")
               for h_i in self.outer_tape.gradient(self.grad_params,
                                                   self.params,
                                                   output_gradients=v)]
        return hvp

    def get_ihvp(self, v):
        """Compute [H_params(self.out)]^(-1)v."""
        assert isinstance(v, list)
        assert len(v) == self.n
        assert self.grad_params is not None and self.outer_tape is not None
        ihvp = v[:]
        for _ in tqdm(range(self.iters), total=self.iters,
                      desc="IterativeIHVP_TF.get_ihvp"):
            # Apply the recursion ihvp <- v + ihvp - H*ihvp
            H_ihvp = self.compute_hvp(ihvp)
            for i in range(self.n):
                ihvp[i] = v[i] + (1.0 - self.reg) * ihvp[i] - H_ihvp[i]
                ihvp[i] = tf.stop_gradient(ihvp[i]) / self.scale
        return ihvp


# class LowRankIHVPTF(BaseIHVPTF):
#
#     """Approximate Hessian-vector products using a low rank approximation
#     of the Hessian.
#
#     The low rank matrix is itself represented using a neural network.
#     """
#
#     class _Net(nn.Module):
#
#         """Network to approximate Hv of the target network."""
#
#         def __init__(self, n_ins, n_h):
#             super().__init__()
#             self.fcs = []
#             for i, n_in in enumerate(n_ins):
#                 fc = nn.Linear(n_in, n_h, bias=False)
#                 setattr(self, f"fc{i}", fc)
#                 self.fcs.append(fc)
#
#         def forward(self, *x):
#             h = sum(fc(x_i) for fc, x_i in zip(self.fcs, x))
#             y = [h @ fc.weight for fc in self.fcs]
#             return y
#
#     # pylint: disable=dangerous-default-value
#     def __init__(
#         self,
#         params,
#         rank,
#         batch_size=np.inf,
#         iters_per_point=20,
#         criterion=F.mse_loss,
#         optim_cls=Adam,
#         optim_kwargs={"lr": 0.01},
#         ckpt_every=1,
#         device=torch.device("cpu"),
#         tb_writer=None,
#     ):
#         super().__init__()
#         self.target_params = list(params)
#         self.numels = [p.numel() for p in self.target_params]
#         self.app_net = self._Net(self.numels, rank).to(device)
#         self.criterion = criterion
#         self.batch_size = batch_size
#         self.iters_per_point = iters_per_point
#         self.optim = optim_cls(params=self.app_net.parameters(), **optim_kwargs)
#         self.total_iter = 0
#         self.n_train = 0
#         self.ckpt_every = ckpt_every
#         self.tb_writer = tb_writer
#         self.P = torch.zeros(sum(self.numels), rank, device=device)
#         self._update_wew()
#
#     def _update_wew(self):
#         """Update the helper tensors used for get_ihvp."""
#         with torch.no_grad():
#             P = torch.cat([fc.weight.t() for fc in self.app_net.fcs], out=self.P)
#             U, S, _ = torch.svd(P)
#             We = U @ torch.diag(1 / S ** 2)
#             Wt = U.t()
#
#             # Convert We and Wt to a list of tensors compatible with the params
#             self.We, self.Wt = [], []
#             n = 0
#             for p, n_i in zip(self.target_params, self.numels):
#                 shape = list(p.shape)
#                 self.We.append(We[n : n + n_i, :].contiguous().view(*(shape + [-1])))
#                 self.Wt.append(Wt[:, n : n + n_i].contiguous().view(*([-1] + shape)))
#                 n += n_i
#
#     def update(self, out, vs):
#         self.out = out
#         train_vs = []
#         for v in vs:
#             train_vs.append([v_i.contiguous().view(-1) for v_i in v])
#         n_new = max(1, len(train_vs) - self.n_train)
#         self.n_train = len(train_vs)
#
#         grad_target_params = [
#             g.contiguous().view(-1)
#             for g in torch.autograd.grad(
#                 self.out, self.target_params, create_graph=True
#             )
#         ]
#
#         idx = 0
#         Y_train = []
#         batch_size = min(self.batch_size, len(train_vs))
#         while idx < self.n_train:
#             batch_X = train_vs[idx : idx + batch_size]
#             idx += len(batch_X)
#             grad_param_vps = [
#                 [
#                     grad_target_params[j].view(-1) @ batch_X[i][j].view(-1)
#                     for j in range(len(self.target_params))
#                 ]
#                 for i in range(len(batch_X))
#             ]
#             with torch.no_grad():
#                 batch_Y = [
#                     torch.cat(
#                         list(
#                             map(
#                                 lambda y: y.contiguous().view(-1).detach(),
#                                 torch.autograd.grad(
#                                     grad_param_vp, self.target_params, create_graph=True
#                                 ),
#                             )
#                         )
#                     )
#                     for grad_param_vp in grad_param_vps
#                 ]
#             Y_train.extend(batch_Y)
#
#         iters = self.iters_per_point * n_new
#         for _ in range(iters):
#             idxs = random.sample(range(self.n_train), batch_size)
#             batch_X = [train_vs[idx] for idx in idxs]
#             batch_Y = [Y_train[idx] for idx in idxs]
#
#             # Compute predictions and loss
#             batch_Yhat = self.app_net(*map(torch.stack, zip(*batch_X)))
#             batch_Yhat = list(zip(*batch_Yhat))
#             loss = sum(
#                 self.criterion(torch.cat(y_hat), y) / batch_size
#                 for y, y_hat in zip(batch_Y, batch_Yhat)
#             )
#             self.total_iter += 1
#             if self.tb_writer is not None and self.total_iter % self.ckpt_every == 0:
#                 self.tb_writer.add_scalar(
#                     "low_rank_ihvp/log_hv_loss",
#                     torch.log10(loss).item(),
#                     self.total_iter,
#                 )
#             self.optim.zero_grad()
#             loss.backward()
#             self.optim.step()
#         self._update_wew()
#
#     def get_ihvp(self, v):
#         assert isinstance(v, list)
#         assert len(v) == len(self.target_params)
#         with torch.no_grad():
#             Wtv = sum(
#                 (
#                     torch.mul(Wt_i, v_i.unsqueeze(0))
#                     .contiguous()
#                     .view(-1, n_i)
#                     .sum(dim=1)
#                 )
#                 for Wt_i, v_i, n_i in zip(self.Wt, v, self.numels)
#             )
#             iHv = [torch.matmul(We_i, Wtv) for We_i in self.We]
#         return iHv
