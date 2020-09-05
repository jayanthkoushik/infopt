import itertools
import math
from argparse import ArgumentParser, FileType
from statistics import mean

# Must be imported before GPy to configure matplotlib
from shinyutils import LazyHelpFormatter

import GPyOpt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from infopt.gpinfacq import GPInfAcq
from infopt.ihvp import LowRankIHVP
from infopt.nnacq import NNAcq
from infopt.nnmodel import NNModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class XORTree:
    def __init__(self, leaf_preds):
        self.leaf_preds = leaf_preds

    def _predict(self, b, p):
        if len(b) == 1:
            return p[0] if b[0] == 0 else p[1]
        hl = len(p) // 2
        if b[0] == 0:
            return self._predict(b[1:], p[:hl])
        return self._predict(b[1:], p[hl:])

    def predict(self, b):
        return self._predict(b, self.leaf_preds)

    @classmethod
    def average_cross_entropy(cls, leaf_preds):
        try:
            if len(leaf_preds.shape) == 2:
                leaf_preds = leaf_preds[0]
        except AttributeError:
            pass

        nbits = int(math.log2(len(leaf_preds)))
        assert 2 ** nbits == len(leaf_preds)

        tree = cls(leaf_preds)

        ces = []
        for b in itertools.product([0, 1], repeat=nbits):
            y = float(sum(b) == 1)
            pred = float(tree.predict(b))
            ce = F.binary_cross_entropy(torch.tensor(pred), torch.tensor(y))
            ces.append(ce.item())
        return mean(ces)


def run(args):
    bo_obj = GPyOpt.core.task.SingleObjective(XORTree.average_cross_entropy)
    bo_space = GPyOpt.Design_space(
        space=[
            {"name": f"leaf_{i}_pred", "type": "continuous", "domain": (0, 1)}
            for i in range(2 ** args.nbits)
        ]
    )
    bo_initial_design = GPyOpt.experiment_design.initial_design(
        "random", bo_space, args.init_samples
    )

    if args.acq.startswith("gp"):
        normalize_Y = True
        bo_model = GPyOpt.models.GPModel(exact_feval=True, ARD=False, verbose=False)
        acq_optimizer = GPyOpt.optimization.AcquisitionOptimizer(bo_space)

        if args.acq == "gp_ei":
            bo_acq = GPyOpt.acquisitions.AcquisitionEI(
                bo_model, bo_space, acq_optimizer
            )
        elif args.acq == "gp_lcb":
            bo_acq = GPyOpt.acquisitions.AcquisitionLCB(
                bo_model, bo_space, acq_optimizer, exploration_weight=2
            )
        else:  # gp_inf
            bo_acq = GPInfAcq(bo_model, bo_space, acq_optimizer, exploration_weight=2)

    else:  # nn_inf
        print(f"device: {DEVICE}")
        normalize_Y = False
        nn_model = nn.Sequential(
            nn.Linear(2 ** args.nbits, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.ReLU(),
        ).to(DEVICE)
        ihvp = LowRankIHVP(
            nn_model.parameters(),
            rank=10,
            optim_cls=optim.Adam,
            optim_kwargs={"lr": 0.01},
            batch_size=16,
            iters_per_point=25,
            device=DEVICE,
        )
        net_optim = optim.Adam(nn_model.parameters(), lr=0.01)
        bo_model = NNModel(
            nn_model,
            ihvp,
            net_optim,
            update_batch_size=32,
            update_iters_per_point=25,
            num_higs=16,
            ihvp_n=32,
            device=DEVICE,
        )
        bo_acq = NNAcq(
            bo_model,
            bo_space,
            exploration_weight=2,
            optim_cls=optim.Adam,
            optim_kwargs={"lr": 0.01},
            optim_iters=1000,
            reinit_optim_start=True,
            lr_decay_step_size=100,
            lr_decay_gamma=0.5,
            device=DEVICE,
        )

    bo_eval = GPyOpt.core.evaluators.Sequential(bo_acq)
    bo_problem = GPyOpt.methods.ModularBayesianOptimization(
        bo_model,
        bo_space,
        bo_obj,
        bo_acq,
        bo_eval,
        bo_initial_design,
        normalize_Y=normalize_Y,
    )

    for _ in range(args.iters):
        bo_problem.run_optimization(max_iter=1, verbosity=False)
        err = float(min(bo_problem.Y))
        if args.res_file:
            print(err, file=args.res_file)
        print(
            f"CE: {float(bo_problem.Y[-1]):0.3g}, "
            f"predictions: {[f'{x:0.3g}' for x in bo_problem.X[-1]]}"
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser(formatter_class=LazyHelpFormatter)
    arg_parser.add_argument("--iters", type=int, default=100)
    arg_parser.add_argument("--nbits", type=int, default=2)
    arg_parser.add_argument("--init-samples", type=int, default=5)
    arg_parser.add_argument(
        "--acq",
        type=str,
        choices=["gp_lcb", "gp_ei", "gp_inf", "nn_inf"],
        required=True,
    )
    arg_parser.add_argument("--res-file", type=FileType("w"))
    _args = arg_parser.parse_args()
    run(_args)
