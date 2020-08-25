"""plotting.py: utilities for plotting results."""

import code
import logging
import os
import pickle
import re
from glob import glob

# Must be imported before GPy to configure matplotlib
from shinyutils.matwrap import MatWrap as mw

import pandas as pd


def load_save_data(res_dir, skip_pats):
    df_rows = []
    for res_file in glob(os.path.join(res_dir, "**", "*.pkl"), recursive=True):
        if any(re.match(pat, res_file) for pat in skip_pats):
            logging.info(f"skipping {res_file}")
            continue
        with open(res_file, "rb") as f:
            res = pickle.load(f)

        mname = res["args"]["mname"].upper()
        if mname == "NN":
            acq_type = "INF"
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
                "Time": res["iter_times"][t],
                "y": res["y"][init_points + t].item(),
                **{f"x{j}": x_j for j, x_j in enumerate(res["X"][init_points + t])},
            }
            if "regrets" in res:
                df_row["R"] = res["regrets"][t]
            df_rows.append(df_row)

    data_df = pd.DataFrame(df_rows)
    return data_df


def plot_performance(args, y="R", data_df=None):
    mw.configure(args.context, args.style, args.font)
    if data_df is None:
        data_df = load_save_data(args.res_dir, args.skip_pats)
    models = data_df["Model"].unique()
    n_models = len(models)

    fig = mw.plt().figure()
    ax = fig.add_subplot(111)
    if args.log_scale:
        ax.set_yscale("log", nonposy="mask")
    mw.sns().lineplot(
        x="t",
        y=y,
        hue="Model",
        hue_order=sorted(models, reverse=True),
        data=data_df,
        palette=mw.palette()[:n_models],
        markers=False,
        err_style="band",
        legend="full",
        ax=ax,
    )
    ax.yaxis.tick_right()
    ax.set_xlabel("$t$")
    ax.set_ylabel("${y}_t$")

    if args.interactive:
        code.interact(local={**locals(), **globals()})
        return

    fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)


def plot_timing(args):
    mw.configure(args.context, args.style, args.font)
    data_df = load_save_data(args.res_dir, args.skip_pats)
    data_df = data_df[data_df["t"] % args.model_update_interval == 0]
    models = data_df["Model"].unique()
    n_models = len(models)

    fig = mw.plt().figure()
    ax = fig.add_subplot(111)
    if args.log_scale:
        ax.set_yscale("log", nonposy="clip")

    if args.interactive:
        code.interact(local={**locals(), **globals()})
        return

    mw.sns().lineplot(
        x="t",
        y="Time",
        hue="Model",
        hue_order=sorted(models, reverse=True),
        data=data_df,
        palette=mw.palette()[:n_models],
        markers=False,
        err_style="band",
        legend="full",
        ax=ax,
    )
    ax.yaxis.tick_right()
    ax.set_xlabel("$t$")
    ax.set_ylabel("Iteration time (s)")
    fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)
