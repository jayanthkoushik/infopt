from argparse import FileType
from functools import partial
from glob import glob
import logging
import os
import pickle
import re
import sys
from uuid import uuid4

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    comma_separated_ints,
    KeyValuePairsType,
    LazyHelpFormatter,
    MatWrap as mw,
    OutputDirectoryType,
    shiny_arg_parser as arg_parser,
)

import GPyOpt
import numpy as np
import torch
import torch.nn.init as init
import trains

from exputils.models import DEVICE, FCNet, model_gp, model_nn_inf
from exputils.optimization import run_optim
from exputils.parsing import (
    make_gp_parser,
    make_nninf_parser,
    make_optim_parser,
    make_plot_parser,
    TensorboardWriterType,
)
from exputils.plotting import plot_performance, plot_timing


def main():
    """Entry point."""
    sub_parsers = arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    run_parser = sub_parsers.add_parser("run", formatter_class=LazyHelpFormatter)
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "--no-trains", action="store_false", dest="trains", default=True
    )
    run_parser.add_argument("--save-dir", type=OutputDirectoryType(), required=True)
    run_parser.add_argument(
        "--tb-dir", type=TensorboardWriterType(), dest="tb_writer", default=None
    )

    obj_parser = run_parser.add_argument_group("objective network")
    obj_parser.add_argument("--obj-in-dim", type=int, default=1, required=True)
    obj_parser.add_argument(
        "--obj-layer-sizes",
        type=comma_separated_ints,
        required=True,
        metavar="int,int[,...]",
    )
    obj_parser.add_argument("--load-target", type=FileType("rb"), default=None)
    obj_parser.add_argument("--w-init-fun", type=str, default=None)
    obj_parser.add_argument(
        "--w-init-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(),
    )
    obj_parser.add_argument("--b-init-fun", type=str, default=None)
    obj_parser.add_argument(
        "--b-init-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(),
    )
    obj_parser.add_argument("--obj-bounds", type=float, nargs=2, default=[-5, 5])
    obj_parser.add_argument(
        "--no-save-target-model",
        action="store_false",
        dest="save_target_model",
        default=True,
    )

    objopt_parser = run_parser.add_argument_group("objective optimization")
    objopt_parser.add_argument(
        "--no-obj-opt", action="store_false", dest="obj_opt", default=True
    )
    objopt_parser.add_argument("--obj-opt-span-n", type=int, default=1000)
    objopt_parser.add_argument("--obj-opt-batch-size", type=int, default=None)

    make_optim_parser(run_parser)

    run_sub_parsers = run_parser.add_subparsers(dest="mname")
    run_sub_parsers.required = False

    gp_parser = run_sub_parsers.add_parser("gp", formatter_class=LazyHelpFormatter)
    make_gp_parser(gp_parser)

    nninf_parser = run_sub_parsers.add_parser("nn", formatter_class=LazyHelpFormatter)
    nninf_parser.add_argument(
        "--model-layer-sizes",
        type=comma_separated_ints,
        required=True,
        metavar="int,int[,...]",
    )
    make_nninf_parser(nninf_parser)

    plotf_parser = sub_parsers.add_parser(
        "plot-target-1d", formatter_class=LazyHelpFormatter
    )
    plotf_parser.add_argument("--x-bounds", type=float, nargs=2, default=[-10, 10])
    plotf_parser.add_argument("--x-span-n", type=int, default=1000)
    plotf_parser.add_argument("--batch-size", type=int, default=None)
    plotf_parser.set_defaults(func=plot_target_1d)
    make_plot_parser(plotf_parser)

    ploto_parser = sub_parsers.add_parser(
        "plot-objectives", formatter_class=LazyHelpFormatter
    )
    ploto_parser.set_defaults(func=partial(plot_performance, y="y"))
    make_plot_parser(ploto_parser)

    plott_parser = sub_parsers.add_parser(
        "plot-timing", formatter_class=LazyHelpFormatter
    )
    plott_parser.set_defaults(func=plot_timing)
    make_plot_parser(plott_parser)
    plott_parser.add_argument("--model-update-interval", type=int, default=1)

    args = arg_parser.parse_args()
    args.func(args)


def run(args):
    if args.trains:
        fstr = f"{args.obj_in_dim},{','.join(args.obj_layer_sizes)}"
        task_name = f"[{str(uuid4())[:8]}] {fstr}: {args.mname}"
        if args.mname == "gp":
            task_name += f"-{args.acq_type}"

        trains.Task.init(project_name="inf-opt nnet_optim", task_name=task_name)
    else:
        logging.warning("trains logging not enabled")

    if args.load_target is not None:
        logging.info(f"loading target from `{args.load_target}`")
        tnet = FCNet(args.obj_in_dim, args.obj_layer_sizes).to(DEVICE)
        map_location = None if DEVICE.type == "cuda" else "cpu"
        tnet.load_state_dict(
            torch.load(args.load_target.name, map_location=map_location)
        )
        obj = TargetNetObj(tnet=tnet)
    else:
        p_inits = []
        for p_init_fname, p_init_params in zip(
            [args.w_init_fun, args.b_init_fun], [args.w_init_params, args.b_init_params]
        ):
            if p_init_fname is None:
                p_inits.append(None)
                continue

            if not p_init_fname.endswith("_"):
                logging.critical(
                    f"invalid init function: {p_init_fname}: "
                    f"should be in-place (ending in '_')"
                )
                sys.exit(1)

            try:
                p_init_f = getattr(init, p_init_fname)
            except AttributeError:
                logging.critical(f"unknown init function: {p_init_fname}")
                sys.exit(1)
            p_inits.append(partial(p_init_f, **p_init_params))

        logging.info(f"w_init: {p_inits[0]}")
        logging.info(f"b_init: {p_inits[1]}")

        obj = TargetNetObj(
            args.obj_in_dim, args.obj_layer_sizes, p_inits[0], p_inits[1]
        )
        tnet = obj.tnet

        if args.save_target_model:
            torch.save(tnet.state_dict(), os.path.join(args.save_dir, "tnet.pt"))

    if args.obj_opt and args.obj_in_dim == 1:
        x_span = torch.linspace(*args.obj_bounds, args.obj_opt_span_n).unsqueeze(1)
        y_span = predict_batchly(tnet, x_span, args.obj_opt_batch_size)
        obj.fmin = y_span.min().item()

    bounds = [
        {"name": f"x_{i}", "type": "continuous", "domain": args.obj_bounds}
        for i in range(args.obj_in_dim)
    ]
    space = GPyOpt.Design_space(bounds)

    if args.mname == "gp":
        model, acq = model_gp(space, args)
        normalize_Y = True
    elif args.mname == "nn":
        acq_fast_project = lambda x: x.clamp_(*args.obj_bounds)
        base_model = FCNet(args.obj_in_dim, args.model_layer_sizes).to(DEVICE)
        logging.debug(base_model)
        model, acq = model_nn_inf(base_model, space, args, acq_fast_project)
        normalize_Y = False

    if not args.mname:
        logging.warning(f"model not specified: no optimization performed")
        result = dict()
    else:
        result = run_optim(obj, space, model, acq, normalize_Y, args)
        del result["bo"]

    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer
    if args.load_target is not None:
        args.load_target.close()
    del args.load_target
    result["fmin"] = obj.fmin
    result["args"] = vars(args)
    with open(os.path.join(args.save_dir, "save_data.pkl"), "wb") as f:
        pickle.dump(result, f)


def plot_target_1d(args):
    fig = mw.plt().figure()
    ax = fig.add_subplot(111)
    if args.log_scale:
        ax.set_yscale("log", nonposy="mask")

    for res_file in glob(os.path.join(args.res_dir, "**", "*.pkl"), recursive=True):
        if any(re.match(pat, res_file) for pat in args.skip_pats):
            logging.info(f"skipping {res_file}")
            continue
        with open(res_file, "rb") as f:
            res = pickle.load(f)["args"]

        if not res["save_target_model"] or res["obj_in_dim"] != 1:
            logging.info(f"skipping {res_file}")
            continue

        target_state_dict_path = os.path.join(res["save_dir"], "tnet.pt")
        tnet = FCNet(res["obj_in_dim"], res["obj_layer_sizes"]).to(DEVICE)
        map_location = None if DEVICE.type == "cuda" else "cpu"
        tnet.load_state_dict(
            torch.load(target_state_dict_path, map_location=map_location)
        )

        x_span = torch.linspace(*args.x_bounds, args.x_span_n).unsqueeze(1)
        y_span = predict_batchly(tnet, x_span, args.batch_size)
        tname = ",".join(map(str, res["obj_layer_sizes"]))
        ax.plot(x_span[:, 0].tolist(), y_span[:, 0].tolist(), label=tname)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)


class TargetNetObj:

    """Objective function wrapper over TargetNet."""

    def __init__(
        self, in_dim=None, layer_sizes=None, w_init=None, b_init=None, tnet=None
    ):
        self.fmin = None
        if tnet is not None:
            self.tnet = tnet
            return

        if in_dim is None or layer_sizes is None:
            raise RuntimeError(
                "in_dim and layer_sizes can't be None when tnet isn't passed"
            )
        self.tnet = FCNet(in_dim, layer_sizes).to(DEVICE)

        if w_init is None and b_init is None:
            logging.info("not overriding default target initialization")
            return

        for layer in self.tnet.layers + [self.tnet.fc_last]:
            if w_init is not None:
                w_init(layer.weight)
            if b_init is not None:
                b_init(layer.bias)

    def f(self, x):
        x = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
        y = self.tnet(x)
        return y.detach().cpu().numpy()


def predict_batchly(net, X, batch_size=None):
    batch_size = len(X) if batch_size is None else batch_size
    i = 0
    Y = []
    while i < len(X):
        X_batch = X[i : i + batch_size].to(DEVICE)
        Y_batch = net(X_batch)
        Y.append(Y_batch)
        i += len(X_batch)
    return torch.cat(Y)


if __name__ == "__main__":
    main()
