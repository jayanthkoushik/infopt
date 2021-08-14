"""exp_bpnn_optim.py: black-box optimization for finding low-energy molecules.

Uses the Behler-Parrinello Neural Network (BPNN).

Usage:
    python exp_bpnn_optim.py run --seed $i \
        --save-file "results/bpnn_ZrO2/nngreedy_$i.pkl" --tb-dir "logdir/bpnn_ZrO2/nngreedy_$i" \
        --n-search 1200 --n-test 500 --exp-multiplier 0.0 \
        --init-points 200 --optim-iters 200 --model-update-interval 8 \
        nninf --layer-sizes 64,64 --activation relu \
        --bom-optim-params "learning_rate=0.005" --bom-weight-decay 1e-4 \
        --bom-up-iters-per-point 50 --bom-up-upsample-new 0.25 \
        --pretrain-epochs 50

    python exp_bpnn_optim.py plot-regrets --res-dir results/bpnn_synthetic \
    --save-file results/bpnn_synthetic/regrets.pdf
    python exp_bpnn_optim.py plot-timing --res-dir results/bpnn_synthetic \
    --save-file results/bpnn_synthetic/timing.pdf
"""

import logging
import os
import pickle

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    CommaSeparatedInts,
    LazyHelpFormatter,
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
from exputils.objectives import BPNNBandit
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

    bpnn_parser = run_parser.add_argument_group("bpnn bandit")
    bpnn_parser.add_argument("--n-search", type=int, default=500)
    bpnn_parser.add_argument("--n-test", type=int, default=500)
    bpnn_parser.add_argument("--seed", type=int, default=0)
    bpnn_parser.add_argument("--acq-n-candidates", type=int, default=np.inf)
    bpnn_parser.add_argument("--acq-batch-size", type=int, default=None)

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
        mname_parser.add_argument("--activation", type=str, default="tanh")
        mname_parser.add_argument("--dropout", type=float, default=0.0)
        mname_parser.add_argument("--dtype", type=tf.dtypes.DType,
                                  default=tf.float32)
        mname_parser.add_argument("--pretrain-batch-size", type=int,
                                  default=None)
        mname_parser.add_argument("--pretrain-epochs", type=int, default=20)
        make_nn_tf2_parser(mname_parser,
                           "nnmcd_tf2" if _mname == "nnmcd" else "nn_tf2")

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

    args = base_arg_parser.parse_args()
    if os.path.exists(args.save_file.name) and os.path.getsize(args.save_file.name) > 0:
        logging.info("save file %s exists, skipping", args.save_file.name)
    else:
        args.func(args)


def run(args):
    """Run optimization."""

    # Setup
    problem = BPNNBandit(
        n_search=args.n_search,
        n_test=args.n_test,
        rng=args.seed,
    )
    logging.info("loaded BPNN data with a %d-%d-%d pretrain-search-test split",
                 problem.n_pretrain, problem.n_search, problem.n_test)

    # Pre-train the model
    base_model = make_bpnn(
        input_dim=problem.input_dim,
        layer_sizes=args.layer_sizes,
        activation=args.activation,
        dropout=args.mcd_dropout if args.mname == "nnmcd" else args.dropout,
        dtype=args.dtype,
    )
    base_model.summary(print_fn=logging.info)
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

    mu_mins, sig_mins, acq_mins, exp_w = [], [], [], []

    def eval_hook(n, bo, postfix_dict):
        nonlocal mu_mins, sig_mins, acq_mins, exp_w
        mu_min, sig_min = bo.model.predict(bo.X[-1:])
        _exp_w = bo.acquisition.exploration_weight
        acq_min = -mu_min + _exp_w * sig_min
        mu_mins.append(mu_min.item())
        sig_mins.append(sig_min.item())
        acq_mins.append(acq_min.item())
        exp_w.append(_exp_w)
        postfix_dict["μ*"] = mu_mins[-1]
        postfix_dict["σ*"] = sig_mins[-1]
        postfix_dict["α*"] = acq_mins[-1]

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
    result.update({
        "mu_mins": mu_mins,
        "sig_mins": sig_mins,
        "acq_mins": acq_mins,
        "exp_w": exp_w,
    })
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
    """Pretrain the base BPNN model using the pretraining data."""

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
