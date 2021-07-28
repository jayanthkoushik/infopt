"""
Plot offline results

Usage:
    python plot_offline.py plot --res-dir results/ackley5d_offline \
        --save-file results/ackley5d_offline/dummy.pdf --y all
    python plot_offline.py plot --res-dir results/ackley5d_offline \
        --save-file results/ackley5d_offline/MSE_test.pdf --y MSE_test
    python plot_offline.py plot --res-dir results/nnet_5x64_5d_offline \
        --save-file results/nnet_5x64_5d_offline/all.pdf --y all \
        --target nnet --models gp_lcb nn_same nn_small
    python plot_offline.py plot --res-dir results/nnet_7layers_25d_offline \
        --save-file results/nnet_7layers_25d_offline/all.pdf --y all \
        --target nnet --models gp_lcb nn_same nn_small
"""

import code
import itertools as it
import logging
import os.path
import pickle

from shinyutils import LazyHelpFormatter, OutputFileType

# Must be imported before GPy to configure matplotlib
from shinyutils.matwrap import MatWrap as mw

import pandas as pd
from exputils.parsing import base_arg_parser, make_plot_parser


def load_offline_data(
        res_dir="results/ackley5d_offline",
        models=("gp_lcb", "gp_inf",
                "nn_small", "nn_large", "nn_wide", "nn_deep"),
        n_data=(1, 10, 100, 1000, 10000),
        n_repeat=10,
        t=0,  # which time index to track after offline training
        target="sim",
) -> pd.DataFrame:

    mnames = {
        "gp_lcb": "GP LCB",
        "gp_inf": "GP INF",
        "nn_small": "NN INF (small)",
        "nn_large": "NN INF (large)",
        "nn_wide": "NN INF (wide)",
        "nn_deep": "NN INF (deep)",
        "nn_deep_lr0.002": "NN INF (deep, lr0.002)",
        "nn_deep_dropout": "NN INF (deep, dropout)",
        "nn_deep_wd": "NN INF (deep, wd1e-4)",
        "nn_same": "NN INF (same)"
    }

    df_rows = []
    for model, n, i in it.product(models, n_data, range(1, n_repeat + 1)):
        try:
            if target == "sim":
                res_file = os.path.join(res_dir, f"{model}_{n}_{i:02d}.pkl")
            elif target == "nnet":
                res_file = os.path.join(res_dir, f"{model}_{n}_{i:02d}",
                                        "save_data.pkl")
            else:
                raise ValueError(f"unrecognized target {target}")
            with open(res_file, "rb") as f:
                res = pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"file not found: {model}, {n}, {i}")
            continue
        except EOFError:
            logging.warning(f"eof: {model}, {n}, {i}")
            continue
        df_row = {
            "Model": mnames[model],
            "N": n,
            "R": res["regrets"][t],
        }
        try:
            df_row.update({
                "MSE_train": res["mse"][t],
                "MSE_test": res["test_mse"][t],
                "mu_min": res["mu_mins"][t],
                "sig_min": res["sig_mins"][t],
                "acq_min": res["acq_mins"][t],
                "y_min": res["y_mins"][t],
                "y_min_test": res["Y_test"].min(),
                "mu_at_min": res["mu_at_mins"][t],
                "sig_at_min": res["sig_at_mins"][t],
                "acq_at_min": res["acq_at_mins"][t],
            })
        except KeyError:
            logging.warning(f"{res_file} does not have offline stats")
        df_rows.append(df_row)

    return pd.DataFrame(df_rows)


def plot_offline(
        args,
        x="N",
        y="MSE",
        data_df=None,
):
    mw.configure(args.plotting_context, args.plotting_style, args.plotting_font)
    if data_df is None:
        data_df = load_offline_data(args.res_dir)
    models = data_df["Model"].unique()
    n_models = len(models)

    fig = mw.plt().figure()
    ax = fig.add_subplot(111)
    if x == "N":
        ax.set_xscale("log", nonposx="mask")
    if args.log_scale:
        ax.set_yscale("log", nonposy="mask")
    mw.sns().lineplot(
        x=x,
        y=y,
        hue="Model",
        hue_order=sorted(models, reverse=True),
        data=data_df,
        palette=mw.palette()[:n_models],
        markers=True,
        err_style="band",
        legend="full",
        ax=ax,
    )
    ax.yaxis.tick_right()
    ax.set_xlabel(r"$N$" if x == "N" else x)
    y_print = {
        "MSE_train": "MSE (train)",
        "MSE_test": "MSE (test)",
        "mu_min": r"$\mu(\hat{x})$",
        "sig_min": r"$\sigma(\hat{x})$",
        "acq_min": r"$\alpha(\hat{x})$",
        "y_min": r"$\min (y)$",
        "y_min_test": r"$\min (y_\mathrm{test})$",
        "mu_at_min": r"$\mu(x^*)$",
        "sig_at_min": r"$\sigma(x^*)$",
        "acq_at_min": r"$\alpha(x^*)$",
    }
    ax.set_ylabel(y_print[y])

    if args.interactive:
        code.interact(local={**locals(), **globals()})
        return

    if args.fig_size:
        fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)
    logging.info(f"figure saved at {args.save_file.name}")


def plot_offline_all(args, x="N", data_df=None):
    y_options = ["MSE_train", "MSE_test", "mu_min", "sig_min", "acq_min",
                 "y_min", "mu_at_min", "sig_at_min", "acq_at_min"]
    if data_df is None:
        data_df = load_offline_data(args.res_dir)
    for y in y_options:
        # TODO: don't actually create args.save_file
        dir_name = os.path.dirname(args.save_file.name)
        args.save_file = OutputFileType()(os.path.join(dir_name, f"{y}.pdf"))
        args.log_scale = y.startswith("MSE")
        plot_offline(args, x=x, y=y, data_df=data_df)


def main():
    """Plot regrets and run times."""

    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    plot_parser = sub_parsers.add_parser(
        "plot", formatter_class=LazyHelpFormatter
    )
    plot_parser.set_defaults(func=plot_offline)
    make_plot_parser(plot_parser)
    y_options = ["MSE", "mu_min", "sig_min", "acq_min", "y_min", "y_min_test",
                 "mu_at_min", "sig_at_min", "acq_at_min"]
    plot_parser.add_argument("--y", default="all",
                             choices=set(y_options + ["all"]))
    plot_parser.add_argument("--models", nargs="+",
                             default=("gp_lcb", "gp_inf", "nn_small",
                                      "nn_large", "nn_wide", "nn_deep"))
    plot_parser.add_argument("--target", default="sim", choices={"sim", "nnet"})
    args = base_arg_parser.parse_args()
    print(args)

    # custom load/save routine
    data_df = load_offline_data(res_dir=args.res_dir,
                                models=args.models,
                                target=args.target)
    if args.y == "all":
        plot_offline_all(args, data_df=data_df)
    else:
        args.func(args, y=args.y, data_df=data_df)


if __name__ == "__main__":
    main()
