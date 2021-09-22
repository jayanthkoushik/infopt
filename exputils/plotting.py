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
        try:
            with open(res_file, "rb") as f:
                res = pickle.load(f)
        except EOFError:
            logging.info(f"EOFError in {res_file}, skipping")

        mname = res["args"]["mname"].upper()
        suffix = ["tf2"] if mname.endswith("_TF2") else []
        if mname in ["NN", "NNINF", "NN_TF2"]:
            mname = "NN"
            # handle greedy case (exploration weight == 0)
            if (res["args"]["use_const_exp_w"] == 0.0 or
                    res["args"]["exp_multiplier"] == 0.0):
                acq_type = "Greedy"
            else:
                acq_type = "INF"
            # specify acquisition optimizer if not Adam
            acq_optim_name = res["args"]["acq_optim_cls"].__name__
            if acq_optim_name != "Adam":
                suffix.append(acq_optim_name)
            # specify weight decay if not zero
            if "weight_decay" in res["args"]["bom_optim_params"]:
                wd = res["args"]["bom_optim_params"]["weight_decay"]
                if wd > 0.0:
                    suffix.append(f"wd{wd}")
            elif "bom_weight_decay" in res["args"]:  # TF2
                wd = res["args"]["bom_weight_decay"]
                if wd > 0.0:
                    suffix.append(f"wd{wd}")
        elif mname in ["NNMCD", "NNMCD_TF2"]:
            mname = "NN"
            # handle greedy case (exploration weight == 0)
            if (res["args"]["use_const_exp_w"] == 0.0 or
                    res["args"]["exp_multiplier"] == 0.0):
                acq_type = "Greedy"
            else:
                acq_type = "MCD"
            dropout = res["args"]["mcd_dropout"]
            suffix.append(f"drop{dropout}")
        elif mname == "NR":
            if res["args"]["use_nrif"]:
                acq_type = "INF"
            elif res["args"]["use_original_nr"]:
                acq_type = "ORIG"
            else:
                acq_type = ""
        elif mname == "GP":
            if res["args"]["mcmc"]:
                mname += " (MCMC)"
            acq_type = res["args"]["acq_type"].upper()
        elif mname == "RANDOM":
            mname = "Random"
            acq_type = ""
        else:
            raise ValueError(f"unknown model type: {mname}")
        mname = f"{mname} {acq_type}".strip()
        suffix = ",".join(suffix)
        mname += f" ({suffix})" if suffix else ""

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
                df_row["R"] = res["regrets"][t]  # pointwise regret gap
                df_row["cR"] = sum(res["regrets"][:t])  # cumulative regret
            if "inst_regrets" in res:
                df_row["iR"] = res["inst_regrets"][t]
            if "mu_mins" in res:
                df_row["mu"] = res["mu_mins"][t]
            if "sig_mins" in res:
                df_row["sigma"] = res["sig_mins"][t]
            if "acq_mins" in res:
                df_row["acq"] = res["acq_mins"][t]
            if "mse_acq" in res:  # --diagnostic
                df_row["mse"] = res["mse_acq"][t]
            if "mse_test" in res:  # --diagnostic
                df_row["mse_test"] = res["mse_test"][t]
            df_rows.append(df_row)

    data_df = pd.DataFrame(df_rows)
    return data_df


def plot_performance(args, y="R", data_df=None, y_print="R"):
    mw.configure(args.plotting_context, args.plotting_style, args.plotting_font)
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
    ax.set_ylabel(rf"${y_print}_t$")

    if args.interactive:
        code.interact(local={**locals(), **globals()})
        return

    if args.fig_size:
        fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)


def plot_timing(args, data_df=None):
    mw.configure(args.plotting_context, args.plotting_style, args.plotting_font)
    if data_df is None:
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
    if args.fig_size:
        fig.set_size_inches(*args.fig_size)
    fig.savefig(args.save_file.name)
