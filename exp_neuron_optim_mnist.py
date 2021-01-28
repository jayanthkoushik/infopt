"""exp_neuron_optim_mnist.py: optimize output of a network trained on MNIST.

Train a network on MNIST, and pick the output neuron corresponding to
a particular number. Treating the function from image space to this output
as a black-box function, find the image that maximizes the output.
"""

import itertools
import logging
import os
import pickle
import random
import re
from argparse import FileType
from glob import glob

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    ClassType,
    KeyValuePairsType,
    LazyHelpFormatter,
    OutputDirectoryType,
)

import GPyOpt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from clearml import Task as ClearMLTask
from skimage.io import imsave
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from tqdm import trange

from exputils.models import DEVICE, model_gp, model_nn_inf
from exputils.optimization import run_optim
from exputils.parsing import (
    base_arg_parser,
    make_gp_parser,
    make_nninf_parser,
    make_optim_parser,
    make_plot_parser,
    TensorboardWriterType,
)
from exputils.plotting import plot_performance

NUM_CLASSES = 10
BASE_IMSIZE = 28
N_FEATURES = BASE_IMSIZE * BASE_IMSIZE
DOMAIN = (-1, 1)


def main():
    """Entry point."""
    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    run_parser = sub_parsers.add_parser("run", formatter_class=LazyHelpFormatter)
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "--neuron",
        type=int,
        choices=range(NUM_CLASSES),
        default=None,
        help="output neuron to optimize (chosen randomly if unspecified)",
    )
    run_parser.add_argument("--save-dir", type=OutputDirectoryType(), required=True)
    run_parser.add_argument(
        "--tb-dir", type=TensorboardWriterType(), dest="tb_writer", default=None
    )
    run_parser.add_argument(
        "--dataset-dir", type=OutputDirectoryType(), default=".data/mnist"
    )
    run_parser.add_argument(
        "--no-obj-optimize", action="store_false", dest="obj_optimize", default=True
    )
    run_parser.add_argument("--imsave-every", type=int, default=50)
    run_parser.add_argument(
        "--no-save-obj-net", action="store_false", dest="save_obj_net", default=True
    )

    make_optim_parser(run_parser)

    objtr_parser = run_parser.add_argument_group(
        "initial training of network which will serve as the objective"
    )
    objtr_parser.add_argument("--load-obj-net", type=FileType("rb"), default=None)
    objtr_parser.add_argument("--obj-train-iters", type=int, default=2000)
    objtr_parser.add_argument("--obj-train-ckpt-every", type=int, default=100)
    objtr_parser.add_argument("--obj-train-batch-size", type=int, default=64)
    objtr_parser.add_argument(
        "--obj-train-optim-cls",
        type=ClassType(Optimizer),
        metavar="optimizer",
        default=Adam,
    )
    objtr_parser.add_argument(
        "--obj-train-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.01),
    )
    objtr_parser.add_argument("--obj-train-lr-decay-step-size", type=int, default=500)
    objtr_parser.add_argument("--obj-train-lr-decay-gamma", type=float, default=0.5)

    objopt_parser = run_parser.add_argument_group(
        "optimization of base network to find the ground truth"
    )
    objopt_parser.add_argument("--obj-opt-iters", type=int, default=10000)
    objopt_parser.add_argument("--obj-opt-ckpt-every", type=int, default=200)
    objopt_parser.add_argument("--obj-opt-pad", type=int, default=5)
    objopt_parser.add_argument("--obj-opt-dropout", type=float, default=0.5)
    objopt_parser.add_argument(
        "--obj-opt-optim-cls",
        type=ClassType(Optimizer),
        metavar="optimizer",
        default=Adam,
    )
    objopt_parser.add_argument(
        "--obj-opt-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.5),
    )
    objopt_parser.add_argument("--obj-opt-lr-decay-step-size", type=int, default=1000)
    objopt_parser.add_argument("--obj-opt-lr-decay-gamma", type=float, default=0.1)

    run_sub_parsers = run_parser.add_subparsers(dest="mname")

    gp_parser = run_sub_parsers.add_parser("gp", formatter_class=LazyHelpFormatter)
    make_gp_parser(gp_parser)

    nninf_parser = run_sub_parsers.add_parser("nn", formatter_class=LazyHelpFormatter)
    make_nninf_parser(nninf_parser)
    nninf_parser.add_argument(
        "--no-pretrain", action="store_false", dest="pretrain", default=True
    )
    nninf_parser.add_argument(
        "--no-save-model-net", action="store_false", dest="save_model_net", default=True
    )
    pre_parser = nninf_parser.add_argument_group(
        "pre-training of model network with binary classification (t1 vs. t2)"
    )
    pre_parser.add_argument(
        "--pretrain-t1", type=int, default=0, choices=range(NUM_CLASSES)
    )
    pre_parser.add_argument(
        "--pretrain-t2", type=int, default=9, choices=range(NUM_CLASSES)
    )
    pre_parser.add_argument("--pretrain-iters", type=int, default=1000)
    pre_parser.add_argument("--pretrain-ckpt-every", type=int, default=50)
    pre_parser.add_argument("--pretrain-batch-size", type=int, default=64)
    pre_parser.add_argument(
        "--pretrain-optim-cls",
        type=ClassType(Optimizer),
        metavar="optimizer",
        default=Adam,
    )
    pre_parser.add_argument(
        "--pretrain-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.01),
    )
    pre_parser.add_argument("--pretrain-lr-decay-step-size", type=int, default=500)
    pre_parser.add_argument("--pretrain-lr-decay-gamma", type=float, default=0.1)

    plotr_parser = sub_parsers.add_parser("plot-ys", formatter_class=LazyHelpFormatter)
    plotr_parser.set_defaults(func=plot_ys)
    make_plot_parser(plotr_parser)

    args = base_arg_parser.parse_args()
    if args.clearml_task:
        ClearMLTask.init(
            project_name="infopt_neuron_optim_mnist", task_name=args.clearml_task
        )
        logging.info(
            f"clearml logging enabled under "
            f"'infopt_neuron_optim_mnist/{args.clearml_task}'"
        )
    args.func(args)


def run(args):
    if args.neuron is None:
        logging.info("randomly choosing target neuron")
        args.neuron = random.choice(range(NUM_CLASSES))
    logging.info(f"using target neuron {args.neuron}")

    obj = NeuronObjective(args)

    if args.save_obj_net:
        logging.info("saving objective net")
        torch.save(
            obj.model.state_dict(), os.path.join(args.save_dir, "objective_net.pt")
        )
    else:
        logging.warning("skipping saving of objective net")

    if args.obj_optimize:
        optim_img = obj.get_optim()
        imsave_path = os.path.join(args.save_dir, "obj_opt_img.png")
        imsave(imsave_path, optim_img.cpu().numpy())
        logging.info(f"saved result of objective optimization to {imsave_path}")
    else:
        logging.warning("skipping optimization of objective network")

    if not args.mname:
        return

    bounds = [
        {"name": f"var_{i}", "type": "continuous", "domain": DOMAIN}
        for i in range(N_FEATURES)
    ]
    space = GPyOpt.Design_space(space=bounds)

    if args.mname == "gp":
        bo_model, acq = model_gp(space, args)
        normalize_Y = True
    else:
        acq_fast_project = lambda x: x.clamp_(*DOMAIN)
        net = ModelNet().to(DEVICE)
        if args.pretrain:
            pretrain_model_net_bcls(list(net.children())[:-2], net.n_fc, args)
        else:
            logging.warning("skipping pretraining")
        bo_model, acq = model_nn_inf(net, space, args, acq_fast_project)
        normalize_Y = False

    def eval_hook(n, bo, postfix_dict):
        # pylint: disable=unused-argument
        nonlocal args

        if (n % args.imsave_every == 0) or (n == args.optim_iters):
            # Save the last queried image
            if args.mname == "nn":
                img = f2img(bo.acquisition.x).detach().cpu().numpy()[0, 0]
            else:
                img = bo.X[-1].reshape(BASE_IMSIZE, BASE_IMSIZE)

            img_save_path = os.path.join(args.save_dir, f"query_img_{n}.png")
            imsave(img_save_path, img)
            if args.tb_writer is not None:
                args.tb_writer.add_image(
                    "query_img", torch.from_numpy(img).unsqueeze(0), n
                )

    result = run_optim(obj, space, bo_model, acq, normalize_Y, args, eval_hook)

    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer
    if args.load_obj_net is not None:
        args.load_obj_net.close()
    del args.load_obj_net
    del result["bo"]
    result["args"] = vars(args)
    with open(os.path.join(args.save_dir, "save_data.pkl"), "wb") as f:
        pickle.dump(result, f)

    if args.mname == "nn" and args.save_model_net:
        torch.save(net.state_dict(), os.path.join(args.save_dir, "model_net.pt"))


def plot_ys(args):
    df_rows = []
    for res_file in glob(os.path.join(args.res_dir, "**", "*.pkl"), recursive=True):
        if any(re.match(pat, res_file) for pat in args.skip_pats):
            logging.info(f"skipping {res_file}")
            continue
        with open(res_file, "rb") as f:
            res = pickle.load(f)

        mname = res["args"]["mname"].upper()
        if mname == "NN":
            mname += " INF"
            if res["args"]["pretrain"]:
                t1 = res["args"]["pretrain_t1"]
                t2 = res["args"]["pretrain_t2"]
                mname += f" (pretrained for {t1} vs. {t2})"
            else:
                mname += " (no pretraining)"
        else:
            if res["args"]["mcmc"]:
                mname += " (MCMC)"
            acq_type = res["args"]["acq_type"].upper()
            mname += f" {acq_type}"

        init_points = res["args"]["init_points"]
        iters = res["args"]["optim_iters"]

        for t in range(iters):
            df_row = {
                "Model": mname,
                "t": t,
                "y": -res["y"][: init_points + t + 1].min().item(),
            }
            df_rows.append(df_row)

    data_df = pd.DataFrame(df_rows)
    plot_performance(args, "y", data_df)


def f2img(x, imsize=BASE_IMSIZE):
    x = x.view(x.shape[0], 1, imsize, imsize)
    return x


class NeuronObjective:

    """Objective function based on neuron output."""

    class Model(nn.Module):

        """Internal network for modeling MNIST."""

        # TODO: make network configurable.

        def __init__(self):
            super().__init__()
            imsize = BASE_IMSIZE

            self.conv1 = nn.Conv2d(1, 8, 5)
            self.bn1 = nn.BatchNorm2d(8)
            self.pool1 = nn.MaxPool2d(2)
            imsize = (imsize - 4) // 2

            self.conv2 = nn.Conv2d(8, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool2 = nn.MaxPool2d(2)
            imsize = (imsize - 4) // 2

            self.n_fc = imsize * imsize * 16

            self.drop1 = nn.Dropout(0.25)
            self.fc1 = nn.Linear(self.n_fc, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            # pylint: disable=arguments-differ
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, self.n_fc)
            x = torch.relu(self.fc1(self.drop1(x)))
            return self.fc2(x)

        def predict(self, x):
            """Get final predictions for a batch."""
            p = torch.softmax(self.forward(x), dim=1)
            return p.max(dim=1)[1]

    def __init__(self, args):
        self.args = args
        self.tb_writer = args.tb_writer
        self.model = self.Model().to(DEVICE)
        self.neuron = args.neuron

        if args.load_obj_net is not None:
            logging.info(f"loading objective from {args.load_obj_net.name}")
            map_location = None if DEVICE.type == "cuda" else "cpu"
            self.model.load_state_dict(
                torch.load(args.load_obj_net.name, map_location=map_location)
            )
        else:
            logging.info("training objective network")
            self._train_model()
        self.fmin = None

    def _train_model(self):
        dataset_train = MNIST(
            root=self.args.dataset_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        dataset_test = MNIST(
            root=self.args.dataset_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        optimizer = self.args.obj_train_optim_cls(
            self.model.parameters(), **self.args.obj_train_optim_params
        )
        lr_sched = StepLR(
            optimizer,
            self.args.obj_train_lr_decay_step_size,
            self.args.obj_train_lr_decay_gamma,
        )

        self.model.train()
        train_loader = DataLoader(
            dataset_train,
            shuffle=True,
            drop_last=True,
            batch_size=self.args.obj_train_batch_size,
            pin_memory=True,
        )
        batch_iter = iter(train_loader)

        test_acc = -1
        with trange(self.args.obj_train_iters, desc="Training MNIST model") as pbar:
            for iter_ in pbar:
                try:
                    X, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_loader)
                    X, y = next(batch_iter)
                X, y = X.to(DEVICE), y.to(DEVICE)

                yhat = self.model(X)
                loss = F.cross_entropy(yhat, y)

                optimizer.zero_grad()
                loss.backward()
                lr_sched.step()
                optimizer.step()

                _n = iter_ + 1
                if (_n % self.args.obj_train_ckpt_every == 0) or (
                    _n == self.args.obj_train_iters
                ):
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(
                            "neuron_train/train_loss", loss.item(), iter_
                        )

                    test_loader = DataLoader(
                        dataset_test,
                        self.args.obj_train_batch_size,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True,
                    )
                    corrects = 0.0
                    self.model.eval()
                    for X, y in test_loader:
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        yhat = self.model.predict(X)
                        corrects += (y == yhat).sum().item()
                    self.model.train()

                    test_acc = corrects / len(dataset_test)
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(
                            "neuron_train/test_acc", test_acc, iter_
                        )

                pbar.set_postfix(loss=float(loss), test_acc=test_acc)

        logging.info(f"final test accuracy: {test_acc:.2%}")
        self.model.eval()

    def f(self, x):
        """Actual objective: f(x)."""
        x = torch.from_numpy(x).to(DEVICE, torch.float)
        x = f2img(x)
        y = -self.model(x)[:, self.neuron]
        return y.detach().cpu().numpy()

    def get_optim(self):
        """Get the image which minimizes f."""
        ihw = BASE_IMSIZE + self.args.obj_opt_pad
        x = torch.randn(1, 1, ihw, ihw, device=DEVICE, requires_grad=True)
        x.data.clamp_(*DOMAIN)
        optimizer = self.args.obj_opt_optim_cls([x], **self.args.obj_opt_optim_params)
        lr_sched = StepLR(
            optimizer,
            self.args.obj_opt_lr_decay_step_size,
            self.args.obj_opt_lr_decay_gamma,
        )
        cr = torch.LongTensor(2)
        self.fmin = float("inf")

        with trange(self.args.obj_opt_iters, desc="Optimizing neuron") as pbar:
            for iter_ in pbar:
                cr.random_(0, self.args.obj_opt_pad + 1)
                x_cr = x[:, :, cr[0] : cr[0] + BASE_IMSIZE, cr[1] : cr[1] + BASE_IMSIZE]
                x_cr = (
                    F.dropout(x_cr, self.args.obj_opt_dropout)
                    * self.args.obj_opt_dropout
                )

                y = self.model(x_cr)[0, self.neuron]
                loss = -y

                optimizer.zero_grad()
                loss.backward()
                lr_sched.step()
                optimizer.step()

                x.data.clamp_(*DOMAIN)

                _n = iter_ + 1
                if (_n % self.args.obj_opt_ckpt_every == 0) or (
                    _n == self.args.obj_opt_iters
                ):
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("neuron_opt/obj", loss.item(), iter_)
                    if self.tb_writer is not None:
                        self.tb_writer.add_image("neuron_opt/opt_img", x[0], _n)
                pbar.set_postfix(loss=float(loss))
                self.fmin = min(self.fmin, float(loss))

        return x.detach()[0, 0]


class ModelNet(nn.Module):

    """Network for modeling data."""

    # TODO: make network configurable.

    def __init__(self):
        super().__init__()
        imsize = BASE_IMSIZE

        self.conv1 = nn.Conv2d(1, 1, 5)
        self.pool1 = nn.MaxPool2d(2)
        imsize = (imsize - 4) // 2

        self.conv2 = nn.Conv2d(1, 1, 5)
        self.pool2 = nn.MaxPool2d(2)
        imsize = (imsize - 4) // 2

        self.n_fc = imsize * imsize * 1
        self.fc = nn.Linear(self.n_fc, 1)
        self.scale = nn.Linear(1, 1)

    def forward(self, x):
        # pylint: disable=arguments-differ
        x = f2img(x)
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.contiguous().view(x.shape[0], -1)
        x = torch.tanh(self.fc(x))
        x = self.scale(x)
        return x


def pretrain_model_net_bcls(model_layers, model_out_features, args):
    """Pretrain the model network for binary classification (t1/t2) on MNIST."""

    class _MNISTb(Dataset):

        """Binary MNIST."""

        def __init__(self, t1, t2, dataset_root, train=True):
            self._dataset = MNIST(
                root=dataset_root,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
            if train:
                y = self._dataset.train_labels
            else:
                y = self._dataset.test_labels
            self.t1 = t1
            self.t2 = t2
            y_valid = (y == t1) | (y == t2)
            self.n = y_valid.sum()
            self.idx_map = [i for i, v_i in enumerate(y_valid) if v_i == 1]
            assert len(self.idx_map) == self.n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            _idx = self.idx_map[idx]
            X, _y = self._dataset[_idx]
            assert _y in (self.t1, self.t2)
            return X, torch.tensor(_y == self.t1, dtype=torch.long)

    class ModelWrapper(nn.Module):

        """Wrapper around the target model, to do binay clasification."""

        def __init__(self, layers, out_features):
            super().__init__()
            self.layers = layers
            self.fc_final = nn.Linear(out_features, 1)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = x.contiguous().view(x.shape[0], -1)
            x = self.fc_final(x)
            return x

    model_wrapper = ModelWrapper(model_layers, model_out_features).to(DEVICE)
    optimizer = args.pretrain_optim_cls(
        itertools.chain(
            *(layer.parameters() for layer in model_layers),
            model_wrapper.fc_final.parameters(),
        ),
        **args.pretrain_optim_params,
    )
    lr_sched = StepLR(
        optimizer, args.pretrain_lr_decay_step_size, args.pretrain_lr_decay_gamma
    )

    t1, t2 = args.pretrain_t1, args.pretrain_t2
    dataset_train = _MNISTb(t1, t2, args.dataset_dir, train=True)
    dataset_test = _MNISTb(t1, t2, args.dataset_dir, train=False)

    train_loader = DataLoader(
        dataset_train,
        args.pretrain_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test,
        args.pretrain_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    batch_iter = iter(train_loader)
    acc = -1
    with trange(args.pretrain_iters, desc="Pretraining model net") as pbar:
        for iter_ in pbar:
            try:
                X, y = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                X, y = next(batch_iter)
            X, y = X.to(DEVICE), y.to(DEVICE)

            yhat = model_wrapper(X)[:, 0]
            loss = F.binary_cross_entropy_with_logits(yhat, y.float())

            optimizer.zero_grad()
            loss.backward()
            lr_sched.step()
            optimizer.step()

            _n = iter_ + 1
            if _n % args.pretrain_ckpt_every == 0 or _n == pbar.total:
                if args.tb_writer is not None:
                    args.tb_writer.add_scalar(
                        "model_pretrain/train_loss", loss.item(), _n
                    )

                corrects = 0.0
                for X, y in iter(test_loader):
                    X, y = X.to(DEVICE), y.byte().to(DEVICE)
                    yhat = (torch.sigmoid(model_wrapper(X)[:, 0]) > 0.5).to(
                        dtype=y.dtype
                    )
                    corrects += (yhat == y).sum().item()
                acc = corrects / len(dataset_test)

                if args.tb_writer is not None:
                    args.tb_writer.add_scalar("model_pretrain/test_acc", acc, _n)
            pbar.set_postfix(loss=float(loss), test_acc=float(acc))

    logging.info(f"final pretrain test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
