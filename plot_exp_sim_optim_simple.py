"""plot_exp_sim_optim_simple.py: plot results of run_exp_optim_simple.py.

Slight modification of exputils.plotting.
"""

import numpy as np
import logging
import os
import pickle
import re
from glob import glob

from shinyutils import LazyHelpFormatter

# Must be imported before GPy to configure matplotlib
from shinyutils.matwrap import MatWrap as mw

import pandas as pd

from exputils.plotting import plot_performance, plot_timing
from exputils.parsing import base_arg_parser, make_plot_parser


def load_save_data(res_dir, skip_pats, max_iters=np.inf):
    df_rows = []
    for res_file in glob(os.path.join(res_dir, "**", "*.pkl"), recursive=True):
        if any(re.match(pat, res_file) for pat in skip_pats):
            logging.info(f"skipping {res_file}")
            continue
        with open(res_file, "rb") as f:
            res = pickle.load(f)

        mname = os.path.splitext(os.path.basename(res_file))[0]
        if mname.startswith("gp"):
            mname = "GP-LCB"
        elif mname.startswith("nninftorch") or mname.startswith("nninf_torch"):
            mname = "NN-INF (torch)"
        elif mname.startswith("nninf"):
            mname = "NN-INF"
        elif mname.startswith("nnmcd"):
            mname = "NN-MCD"
        else:
            raise ValueError(f"unknown model type: {mname}")

        if mname == "GP-LCB":
            # res is a BayesianOptimization object...
            bo = res
            init_points = 10 if bo.X.shape[1] > 2 else 1
            iters = min(bo.num_acquisitions, max_iters)
            regrets = np.minimum.accumulate(bo.Y[init_points:])
            for t in range(iters):
                df_row = {
                    "Model": mname,
                    "t": t,
                    "Time": res.cum_time,  # did not keep track :(
                    "y": res.Y[init_points + t].item(),
                    **{f"x{j}": x_j for j, x_j in enumerate(res.X[init_points + t])},
                    "R": regrets[t].item(),
                }
                df_rows.append(df_row)
        else:
            init_points = 1
            iters = min(len(res["iter_times"]), max_iters)
            iter_times = np.cumsum(res["iter_times"])
            for t in range(iters):
                df_row = {
                    "Model": mname,
                    "t": t,
                    "Time": iter_times[t],  # cumulative
                    "y": res["y"][init_points + t].item(),
                    **{f"x{j}": x_j for j, x_j in enumerate(res["X"][init_points + t])},
                    "mu_last": res['μ*'][t],
                    "sig_last": res['σ*'][t],
                    "mu_at_min": res['μ(x*)'][t],
                    "sig_at_min": res['σ(x*)'][t],
                }
                if "regrets" in res:
                    df_row["R"] = res["regrets"][t]
                df_rows.append(df_row)

    data_df = pd.DataFrame(df_rows)
    return data_df


def main():
    """Plot regrets and run times."""

    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    plotr_parser = sub_parsers.add_parser(
        "plot-regrets", formatter_class=LazyHelpFormatter
    )
    plotr_parser.set_defaults(func=plot_performance)
    make_plot_parser(plotr_parser)
    plotr_parser.add_argument("--max-iters", type=int, default=np.inf)

    plott_parser = sub_parsers.add_parser(
        "plot-timing", formatter_class=LazyHelpFormatter
    )
    plott_parser.set_defaults(func=plot_timing)
    make_plot_parser(plott_parser)
    plott_parser.add_argument("--model-update-interval", type=int, default=1)
    plott_parser.add_argument("--max-iters", type=int, default=np.inf)
    args = base_arg_parser.parse_args()
    print(args)  # res_dir, skip_pats=[], save_file

    # custom load/save routine
    data_df = load_save_data(args.res_dir, args.skip_pats, args.max_iters)
    args.func(args, data_df=data_df)


if __name__ == "__main__":
    main()
