"""exp_bpnn_optim.py: black-box optimization for finding low-energy molecules.

Uses the Behler-Parrinello Neural Network (BPNN).

Usage:
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nninf_$i.pkl" \
        --tb-dir "logdir/bpnn_ZrO2/nninf_$i" \
        --init-points 500 --optim-iters 100 --model-update-interval 4 \
        nninf --layer-sizes 64,64 --activation relu \
        --bom-optim-params "learning_rate=0.001" --bom-weight-decay 1e-4 \
        --bom-up-batch-size 64 --bom-up-iters-per-point 25 --bom-up-upsample-new 0.1

    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/random_$i.pkl" \
        --init-points 500 --optim-iters 100 \
        random

    python exp_bpnn_optim.py evaluate-retrieval \
        --res-pats "results/bpnn_ZrO2/nninf*.pkl" \
        --save-file "results/bpnn_ZrO2/nninf_retrieval.csv"
"""

from functools import partial
from glob import glob
import logging
import os
import pickle
import time
from tqdm import trange

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    CommaSeparatedInts,
    LazyHelpFormatter,
    InputFileType,
)
from shinyutils.matwrap import MatWrap as mw

from GPyOpt import Design_space

import numpy as np
import tensorflow as tf

from exputils.optimization import run_optim
from exputils.parsing import (
    base_arg_parser,
    make_optim_parser,
    make_nn_tf2_parser,
    make_plot_parser,
    TensorboardWriterType,
    OutputFileType,
)
from exputils.plotting import plot_performance, plot_timing

# BPNN
from exputils.objectives_bpnn import BPNNBandit
from exputils.models_bpnn import make_bpnn, model_bpnn_inf, model_bpnn_mcd


logging.basicConfig(level=logging.DEBUG)


def main():
    """Entry point."""
    sub_parsers = base_arg_parser.add_subparsers(dest="cmd")
    sub_parsers.required = True

    run_parser = sub_parsers.add_parser("run", formatter_class=LazyHelpFormatter)
    run_parser.set_defaults(func=run)
    run_parser.add_argument("--save-file", type=OutputFileType(), required=True)
    make_optim_parser(run_parser)
    run_parser.add_argument(
        "--tb-dir", type=TensorboardWriterType(), dest="tb_writer", default=None
    )
    # computes MSE over acquired and test points at each step
    run_parser.add_argument("--diagnostic", action="store_true", default=False)

    bpnn_parser = run_parser.add_argument_group("bpnn bandit")
    # Some used for pre-training directly in infopt
    bpnn_parser.add_argument("--input-dim", type=int, default=8)
    bpnn_parser.add_argument("--search-split", type=float, default=1.0)
    bpnn_parser.add_argument("--n-test", type=int, default=500)
    # Controls how pretrain/search is split (not test)
    bpnn_parser.add_argument("--seed", type=int, default=0)
    bpnn_parser.add_argument("--acq-n-candidates", type=int, default=np.inf)
    bpnn_parser.add_argument("--acq-batch-size", type=int, default=256)

    run_sub_parsers = run_parser.add_subparsers(dest="mname")
    run_sub_parsers.required = True
    for _mname in ["nninf", "nnmcd"]:
        mname_parser = run_sub_parsers.add_parser(
            _mname, formatter_class=LazyHelpFormatter
        )
        mname_parser.add_argument(
            "--layer-sizes",
            type=CommaSeparatedInts(),
            required=True,
            metavar="int,int[,...]",
        )
        mname_parser.add_argument("--activation", type=str, default="relu")
        mname_parser.add_argument("--dropout", type=float, default=0.0)
        mname_parser.add_argument("--dtype", type=tf.dtypes.DType,
                                  default=tf.float32)
        mname_parser.add_argument("--pretrain-batch-size", type=int,
                                  default=None)
        mname_parser.add_argument("--pretrain-epochs", type=int, default=20)
        make_nn_tf2_parser(mname_parser,
                           "nnmcd_tf2" if _mname == "nnmcd" else "nn_tf2")
    run_sub_parsers.add_parser(
        "random", formatter_class=LazyHelpFormatter
    )

    plotr_parser = sub_parsers.add_parser(
        "plot-regrets", formatter_class=LazyHelpFormatter
    )
    plotr_parser.set_defaults(func=plot_performance)
    make_plot_parser(plotr_parser)

    plott_parser = sub_parsers.add_parser(
        "plot-timing", formatter_class=LazyHelpFormatter
    )
    plott_parser.set_defaults(func=plot_timing)
    make_plot_parser(plott_parser)
    plott_parser.add_argument("--model-update-interval", type=int, default=1)

    # Additional plotting options
    for y, y_print in [
        ("cR", "cR"), ("iR", r"\Delta"),
        ("mu", r"\mu"), ("sigma", r"\sigma"), ("acq", r"\alpha"),
        ("mse", r"\text{MSE}"), ("mse_test", r"\text{MSE-Test}"),
    ]:
        plotacq_parser = sub_parsers.add_parser(
            f"plot-{y}", formatter_class=LazyHelpFormatter
        )
        plotacq_parser.set_defaults(
            func=partial(plot_performance, y=y, y_print=y_print))
        make_plot_parser(plotacq_parser)

    # Evaluation of acquired points
    plote_parser = sub_parsers.add_parser(
        "evaluate-retrieval", formatter_class=LazyHelpFormatter
    )
    plote_parser.set_defaults(func=evaluate_retrieval)
    plote_parser.add_argument("--res-pats", type=str, required=True)
    plote_parser.add_argument("--save-file", type=OutputFileType())
    plote_parser.add_argument("--beta", type=float, default=1.0)
    plote_parser.add_argument("--cutoff", type=float, default=-0.8)

    args = base_arg_parser.parse_args()
    if (hasattr(args.save_file, "name") and
            os.path.exists(args.save_file.name) and
            os.path.getsize(args.save_file.name) > 0):
        logging.info("save file %s exists, skipping", args.save_file.name)
    else:
        args.func(args)


def run(args):
    """Run optimization."""

    # Setup
    problem = BPNNBandit(
        input_dim=args.input_dim,
        search_split=args.search_split,
        n_test=args.n_test,
        rng=args.seed,
    )
    logging.info("loaded BPNN data with a %d-%d-%d pretrain-search-test split",
                 problem.n_pretrain, problem.n_search, problem.n_test)

    if args.mname == "random":
        run_random_search(problem, args)
        return

    # Pre-train the model
    base_model = make_bpnn(
        input_dim=problem.input_dim,
        layer_sizes=args.layer_sizes,
        activation=args.activation,
        dropout=args.mcd_dropout if args.mname == "nnmcd" else args.dropout,
        dtype=args.dtype,
    )
    base_model.summary(print_fn=logging.info)
    if args.search_split < 1.0:
        logging.warning(f"pretraining does not train the ihvp estimator")
        pretrain_bpnn(problem, base_model, args)

    # Search over held-out examples
    domain = problem.get_domain()
    space = Design_space(domain)
    dtype = args.dtype

    def feature_map(X, Y=None):
        """Map X (one-hot) to `tf.RaggedTensor`s and Y to `tf.Tensor`s."""
        nonlocal problem, dtype

        X_np = problem.get_features(np.expand_dims(np.argmax(X, axis=1), 1))
        return preprocess_ragged_input(X_np, Y, dtype=dtype)

    if args.mname == "nninf":
        model, acq = model_bpnn_inf(base_model, space, args, feature_map)
    elif args.mname == "nnmcd":
        model, acq = model_bpnn_mcd(base_model, space, args, feature_map)
    else:
        raise ValueError(f"unrecognized model name {args.mname}")

    monitors = {
        "mu_mins": [],
        "sig_mins": [],
        "acq_mins": [],
        "exp_w": [],
    }
    diagnostic = args.diagnostic
    if diagnostic:
        monitors.update({
            "mse_acq": [],
            "std_acq": [],
            "mse_test": [],
            "std_test": [],
        })
        X_test, _ = preprocess_ragged_input(
            problem.get_features(problem.X_test), None)

    def eval_hook(n, bo, postfix_dict):
        nonlocal monitors, diagnostic
        _exp_w = bo.acquisition.exploration_weight
        mu, sig = bo.model.predict(bo.X[-1:])
        acq = -mu + _exp_w * sig
        mu, sig, acq = [t.item() for t in [mu, sig, acq]]
        monitors["mu_mins"].append(mu)
        monitors["sig_mins"].append(sig)
        monitors["acq_mins"].append(acq)
        monitors["exp_w"].append(_exp_w)
        postfix_dict["μ*"] = mu
        postfix_dict["σ*"] = sig
        postfix_dict["α*"] = acq
        if diagnostic:
            nonlocal problem, X_test
            # evaluate over acquired points
            X, _ = preprocess_ragged_input(problem.get_features(bo.X), None)
            preds = bo.model.net.predict(X, 256)
            mse_acq = np.mean((preds - bo.Y) ** 2)
            std_acq = np.std(preds)
            monitors["mse_acq"].append(mse_acq)
            monitors["std_acq"].append(std_acq)
            postfix_dict["mse_acq"] = mse_acq
            postfix_dict["std_acq"] = std_acq
            # evaluate over test points
            preds_test = bo.model.net.predict(X_test, 256)
            mse_test = np.mean((preds_test - problem.Y_test) ** 2)
            std_test = np.std(preds_test)
            monitors["mse_test"].append(mse_test)
            monitors["std_test"].append(std_test)
            postfix_dict["mse_test"] = mse_test
            postfix_dict["std_test"] = std_test

    result = run_optim(problem, space, model, acq, False, args, eval_hook)

    save_file = args.save_file
    del args.save_file
    del args.func
    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer
    del result["bo"]
    result["fmin"] = problem.fmin
    result["args"] = vars(args)
    result.update(monitors)
    pickle.dump(result, save_file.buffer)
    save_file.close()


def preprocess_input(x, y=None, dtype=tf.float32):
    """Preprocess a molecule into a batch of atomic features."""
    x_tf = tf.convert_to_tensor([gvec for gvec in x if gvec != "0"],
                                dtype=dtype)
    y_tf = tf.convert_to_tensor(y, dtype=dtype) if y is not None else y
    return x_tf, y_tf


def preprocess_ragged_input(X, Y=None, dtype=tf.float32):
    """Preprocess a batch/set of molecules into a ragged tensor of atomic features.

    Reference:
        https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
    """
    values = [[gvec for gvec in x if gvec != "0"] for x in X]
    row_lengths = [len(gvecs) for gvecs in values]
    row_splits = np.cumsum([0] + row_lengths)
    values = [gvec for gvecs in values for gvec in gvecs]  # flattened
    X_tf = tf.cast(tf.RaggedTensor.from_row_splits(values, row_splits),
                   dtype=dtype)  # tf.FloatTensor(n_mols, None, input_dim)
    Y_tf = tf.convert_to_tensor(Y, dtype=dtype) if Y is not None else Y
    return X_tf, Y_tf


def pretrain_bpnn(problem, model, args):
    """Pretrain the base BPNN model using the pretraining data.

    *Deprecated:* This does not train the ihvp estimator.
    Use --init-points to set the number of pre-training data directly.
    """

    # Pretraining
    X, Y = problem.get_data("pretrain")
    X, Y = preprocess_ragged_input(X, Y, dtype=args.dtype)

    model.compile(
        loss="mse",
        optimizer=args.bom_optim_cls(**args.bom_optim_params),
        metrics=["mse"],
    )
    model.fit(
        X,
        Y,
        batch_size=args.pretrain_batch_size,
        epochs=args.pretrain_epochs,
    )

    # Evaluation before active search
    X_test, Y_test = problem.get_data("test")
    X_test, Y_test = preprocess_ragged_input(X_test, Y_test, dtype=args.dtype)
    test_scores = model.evaluate(X_test, Y_test, verbose=0)
    logging.info("test MSE after pretraining: %.5f", test_scores[1])

    return model


def run_random_search(problem, args):
    """Random search baseline."""

    X, Y = problem.X_search, problem.Y_search
    indices = np.arange(Y.shape[0])
    rng = np.random.default_rng(args.seed + 1)

    init_points_idx = rng.choice(indices, args.init_points)
    X_acq, Y_acq = X[init_points_idx], Y[init_points_idx]

    # Sequentially monitor acquisitions so that we can compare with NN models
    inst_regrets, regrets, iter_times = [], [], []
    y_best = np.inf
    t0 = time.time()
    for _ in trange(args.optim_iters, desc="Random Selection", leave=True):
        idx = rng.choice(indices)
        X_acq = np.vstack([X_acq, X[idx, np.newaxis]])
        Y_acq = np.vstack([Y_acq, Y[idx, np.newaxis]])
        y_acq = Y_acq[-1].item()
        y_best = min(y_acq, y_best)
        if problem.fmin is not None:
            inst_regrets.append(y_acq - problem.fmin)
            regrets.append(y_best - problem.fmin)
        iter_times.append(time.time() - t0)
        t0 = time.time()

    save_file = args.save_file
    del args.save_file
    del args.func
    if args.tb_writer is not None:
        args.tb_writer.close()
    del args.tb_writer

    result = {
        "inst_regrets": inst_regrets,
        "regrets": regrets,
        "X": X_acq,
        "y": Y_acq,
        "iter_times": iter_times,
        "fmin": problem.fmin,
        "args": vars(args),
    }
    pickle.dump(result, save_file.buffer)
    save_file.close()


def evaluate_retrieval(args):
    """Evaluate acquisitions using precision, recall, and f-beta scores."""

    res_files = [res_file for res_file in glob(args.res_pats)
                 if res_file.endswith(".pkl")]

    precisions, recalls, fscores = [], [], []
    for res_file in res_files:
        with open(res_file, "rb") as f:
            result = pickle.load(f)

        problem = BPNNBandit(
            input_dim=result["args"]["input_dim"],
            search_split=result["args"]["search_split"],
            n_test=result["args"]["n_test"],
            rng=result["args"]["seed"],
        )

        X_retrieved = result["X"][result["args"]["init_points"]:]
        precision, recall, fscore = problem.compute_metrics(
            X_retrieved, beta=args.beta, cutoff=args.cutoff, split="search")
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    p_avg, p_std = np.mean(precisions).item(), np.std(precisions).item()
    r_avg, r_std = np.mean(recalls).item(), np.std(recalls).item()
    f_avg, f_std = np.mean(fscores).item(), np.std(fscores).item()

    logging.info("Files:     \t %s (N=%d)", args.res_pats, len(precisions))
    logging.info("Precision: \t %.5f +/- %.5f", p_avg, p_std)
    logging.info("Recall:    \t %.5f +/- %.5f", r_avg, r_std)
    logging.info("F%g score: \t %.5f +/- %.5f", args.beta, f_avg, f_std)

    if args.save_file:
        args.save_file.writelines([
            "Name,Value,StDev\n",
            "FilePattern,%s,\n" % args.res_pats,
            "NFiles,%s,\n" % len(precisions),
            "Beta,%g,\n" % args.beta,
            "Precision,%.5f,%.5f\n" % (p_avg, p_std),
            "Recall,%.5f,%.5f\n" % (r_avg, r_std),
            "FScore,%.5f,%.5f\n" % (f_avg, f_std),
        ])
        args.save_file.close()


if __name__ == "__main__":
    main()


# """
# torch modules (WIP)
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def model_bpnn_inf_torch(base_model, args, space, acq_fast_project=None):
#     """Setup the black-box optimization models (model, acq, ihvp)."""
#     raise NotImplementedError
#
#
# class BPNN(nn.Module):
#     """Behler-Parrinello Neural Network.
#
#     Accepts a variable-length vector of atom features ("g-functions") as input.
#     Outputs the predicted energy (real-valued scalar), which is computed as
#     the sum of atomic energies.
#
#     Args:
#         num_gfeatures (int): number of g-function features per atom (e.g., 8)
#         layer_sizes (:obj:`Iterable[int]`):
#             number of hidden units across layers (e.g., [26, 26])
#         dropout (float): dropout rate (default: 0.0)
#
#     Attributes:
#
#     """
#     def __init__(self, num_gfeatures, layer_sizes,
#                  activation="tanh", dropout=0.1):
#         super().__init__()
#         layer_sizes = [num_gfeatures] + layer_sizes
#         self.layers = nn.ModuleList([
#             nn.Linear(n_in, n_out, bias=True)
#             for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
#         ])
#         self.output_layer = nn.Linear(layer_sizes[-1], 1, bias=True)
#         self.activation = getattr(F, activation)
#         self.dropout = dropout
#
#     def forward(self, x):
#         """Accepts a single molecule of shape (num_atoms, num_gfeatures)."""
#         h = x
#         for layer in self.layers:
#             h = self.activation(layer(h))
#             if self.dropout > 0:
#                 h = F.dropout(h)
#         e = self.output_layer(h)
#         return torch.sum(e)
