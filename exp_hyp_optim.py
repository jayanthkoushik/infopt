"""exp_hyp_optim.py: test algorithms for hyper-parameter optimization."""

import logging
import pickle
from abc import ABCMeta, abstractmethod

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    ClassType,
    CommaSeparatedInts,
    KeyValuePairsType,
    LazyHelpFormatter,
    OutputFileType,
)

import numpy as np
import torch
from clearml import Task as ClearMLTask
from GPyOpt import Design_space
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from exputils.models import DEVICE, FCNet, model_gp, model_nn_inf
from exputils.optimization import run_optim
from exputils.parsing import (
    base_arg_parser,
    make_gp_parser,
    make_nn_nr_parser,
    make_optim_parser,
    TensorboardWriterType,
)


def main():
    """Entry point."""
    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    run_parser = sub_parsers.add_parser("run", formatter_class=LazyHelpFormatter)
    run_parser.set_defaults(func=run)
    run_parser.add_argument("--save-file", type=OutputFileType(), required=True)
    run_parser.add_argument(
        "--tb-dir", type=TensorboardWriterType(), dest="tb_writer", default=None
    )

    obj_parser = run_parser.add_argument_group("objective function")
    obj_parser.add_argument("--obj-cls", type=ClassType(HypOptObj), required=True)
    obj_parser.add_argument(
        "--obj-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(),
    )

    data_parser = run_parser.add_argument_group("dataset")
    data_parser.add_argument("--dataset-cls", type=ClassType(Dataset), required=True)
    data_parser.add_argument(
        "--dataset-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(),
    )

    make_optim_parser(run_parser)

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
        make_nn_nr_parser(mname_parser, _mname)

    args = base_arg_parser.parse_args()
    if args.clearml_task:
        ClearMLTask.init(project_name="infopt_hyp_optim", task_name=args.clearml_task)
        logging.info(
            f"clearml logging enabled under 'infopt_hyp_optim/{args.clearml_task}'"
        )
    args.func(args)


def run(args):
    dataset = args.dataset_cls(**args.dataset_params)
    obj = args.obj_cls(dataset, **args.obj_params)
    space = Design_space(obj.bounds)

    if args.mname == "gp":
        model, acq = model_gp(space, args)
        normalize_Y = True
    else:
        acq_fast_project = obj.project
        if acq_fast_project is None:
            logging.info("using default projection")
        else:
            logging.info(f"using fast project {acq_fast_project}")
        base_model = FCNet(obj.n, args.layer_sizes).to(DEVICE)
        logging.debug(base_model)
        model, acq = model_nn_inf(base_model, space, args, acq_fast_project)
        normalize_Y = False

    result = run_optim(obj, space, model, acq, normalize_Y, args)

    save_file = args.save_file
    del args.save_file
    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer
    del result["bo"]
    result["args"] = vars(args)
    pickle.dump(result, save_file.buffer)
    save_file.close()


class HypOptObj(metaclass=ABCMeta):

    """Base class for objective representing hyper-prarameter optimization."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.fmin = None

    @property
    def n(self):
        """Number of hyper-parameters."""
        return len(self.bounds)

    @property
    @abstractmethod
    def bounds(self):
        """Bounds for hyper-parameter values."""
        raise NotImplementedError

    @property
    def loss(self):
        """Validation loss function."""
        return mean_squared_error

    @property
    def project(self):
        """Function to project points to input space."""
        return None

    @abstractmethod
    def _get_model(self, *x_i):
        raise NotImplementedError

    def _f_single(self, *x_i):
        model = self._get_model(*x_i)
        model.fit(self.dataset.X_tr, self.dataset.y_tr)
        yhat_te = model.predict(self.dataset.X_te)
        return self.loss(yhat_te, self.dataset.y_te)

    def f(self, x):
        """Loss for each set of hyper-parameter values."""
        y = []
        for x_i in x:
            y_i = self._f_single(*x_i)
            y.append(y_i)
        return np.vstack(y)


class SVRHypOptObj(HypOptObj):

    """SVR hyper-parameter optimization."""

    _BOUND = (1e-5, 100)

    @property
    def bounds(self):
        return [
            {"name": "C", "type": "continuous", "domain": self._BOUND},
            {"name": "gamma", "type": "continuous", "domain": self._BOUND},
            {"name": "epsilon", "type": "continuous", "domain": self._BOUND},
        ]

    @property
    def project(self):
        return lambda x: x.clamp_(*self._BOUND)

    def _get_model(self, C, gamma, epsilon):
        return SVR(C=C, gamma=gamma, epsilon=epsilon)


class LassoHypOptObj(HypOptObj):

    """Lasso hyper-parameter optimization."""

    _BOUND = (0, 100)

    @property
    def bounds(self):
        return [{"name": "alpha", "type": "continuous", "domain": self._BOUND}]

    @property
    def project(self):
        return lambda x: x.clamp_(*self._BOUND)

    def _get_model(self, alpha):
        return Lasso(alpha=alpha)


class Dataset(metaclass=ABCMeta):
    """Base class for datasets."""

    def __init__(self, test_split_size=0.2):
        X_y, X_y_te = self._get_data()
        if X_y_te is None:
            self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
                *X_y, test_size=test_split_size
            )
        else:
            self.X_tr, self.y_tr = X_y
            self.X_te, self.y_te = X_y_te

    @abstractmethod
    def _get_data(self):
        raise NotImplementedError


class CalifHousingDataset(Dataset):
    def __init__(self, data_home=None, **kwargs):
        self.data_home = data_home
        super().__init__(**kwargs)

    def _get_data(self):
        return (
            fetch_california_housing(data_home=self.data_home, return_X_y=True),
            None,
        )


class CIFAR10Dataset(Dataset):
    def __init__(self, data_home, **kwargs):
        self.data_home = data_home
        super().__init__(**kwargs)

    def _get_data(self):
        d_tr = CIFAR10(self.data_home, train=True, transform=ToTensor(), download=True)
        X_tr, y_tr = list(*d_tr)
        X_tr, y_tr = torch.stack(X_tr).float(), torch.tensor(y_tr)
        d_te = CIFAR10(self.data_home, train=False, transform=ToTensor(), download=True)
        X_te, y_te = list(*d_te)
        X_te, y_te = torch.stack(X_te), torch.tensor(y_te)
        return (X_tr, y_tr), (X_te, y_te)


if __name__ == "__main__":
    main()
