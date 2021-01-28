"""parsing.py: functions to add argparse arguments for common modules."""

import argparse
import logging

# Must be imported before GPy to configure matplotlib
from shinyutils import (
    build_log_argp,
    ClassType,
    KeyValuePairsType,
    LazyHelpFormatter,
    OutputDirectoryType,
    OutputFileType,
)
from shinyutils.matwrap import MatWrap as mw

from GPyOpt.core.evaluators import Sequential
from GPyOpt.core.evaluators.base import EvaluatorBase
from torch.nn.modules.loss import _Loss, MSELoss
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

base_arg_parser = argparse.ArgumentParser(formatter_class=LazyHelpFormatter)
build_log_argp(base_arg_parser)
base_arg_parser.add_argument(
    "--clearml-task", type=str, help="enable clearml logging under given task name",
)


def make_nninf_parser(parser):
    """Add parser options for nninf."""
    ihvp_parser = parser.add_argument_group("model low-rank ihvp")
    ihvp_parser.add_argument("--ihvp-rank", type=int, default=10)
    ihvp_parser.add_argument("--ihvp-batch-size", type=int, default=8)
    ihvp_parser.add_argument(
        "--ihvp-optim-cls", type=ClassType(Optimizer), metavar="optimizer", default=Adam
    )
    ihvp_parser.add_argument(
        "--ihvp-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.01),
    )
    ihvp_parser.add_argument("--ihvp-ckpt-every", type=int, default=25)
    ihvp_parser.add_argument("--ihvp-iters-per-point", type=int, default=25)
    ihvp_parser.add_argument(
        "--ihvp-loss-cls", type=ClassType(_Loss), metavar="loss", default=MSELoss
    )

    nn_boparser = parser.add_argument_group("model wrapper for gpyopt")
    nn_boparser.add_argument(
        "--bom-optim-cls", type=ClassType(Optimizer), metavar="optimizer", default=Adam
    )
    nn_boparser.add_argument(
        "--bom-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.02),
    )
    nn_boparser.add_argument("--bom-up-batch-size", type=int, default=16)
    nn_boparser.add_argument("--bom-up-iters-per-point", type=int, default=50)
    nn_boparser.add_argument("--bom-n-higs", type=int, default=8)
    nn_boparser.add_argument("--bom-ihvp-n", type=int, default=15)
    nn_boparser.add_argument("--bom-ckpt-every", type=int, default=25)
    nn_boparser.add_argument("--bom-loss-cls", type=ClassType(_Loss), default=MSELoss)

    nnacq_parser = parser.add_argument_group("lcb acquisition")
    nnacq_parser.add_argument(
        "--acq-optim-cls", type=ClassType(Optimizer), metavar="optimizer", default=Adam
    )
    nnacq_parser.add_argument(
        "--acq-optim-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(lr=0.05),
    )
    nnacq_parser.add_argument("--acq-optim-iters", type=int, default=5000)
    nnacq_parser.add_argument("--acq-optim-lr-decay-step-size", type=int, default=1000)
    nnacq_parser.add_argument("--acq-optim-lr-decay-gamma", type=float, default=0.2)
    nnacq_parser.add_argument("--acq-ckpt-every", type=int, default=1000)
    nnacq_parser.add_argument("--acq-rel-tol", type=float, default=1e-3)
    nnacq_parser.add_argument(
        "--no-acq-reinit-optim-start",
        action="store_false",
        dest="acq_reinit_optim_start",
        default=True,
    )


def make_gp_parser(parser):
    """Add parser options for gp models."""
    gp_parser = parser.add_argument_group("gp configuration")
    gp_parser.add_argument(
        "--acq-type", type=str, default="lcb", choices=["lcb", "ei", "mpi", "inf"]
    )
    gp_parser.add_argument("--mcmc", action="store_true")
    gp_parser.add_argument(
        "--no-exact-feval", action="store_false", dest="exact_feval", default=True
    )
    gp_parser.add_argument("--ard", action="store_true")


def make_optim_parser(parser):
    """Add global optimization arguments to parser."""
    optim_parser = parser.add_argument_group("optimization parameters")
    optim_parser.add_argument("--init-points", type=int, default=1)
    optim_parser.add_argument("--optim-iters", type=int, default=1)
    optim_parser.add_argument("--model-update-interval", type=int, default=1)

    exp_parser = parser.add_argument_group("exploration weight configuration")
    exp_parser.add_argument("--exp-multiplier", type=float, default=0.1)
    exp_parser.add_argument("--exp-gamma", type=float, default=0.1)
    exp_parser.add_argument("--use-const-exp-w", type=float, default=None)

    eval_parser = parser.add_argument_group("evaluator for selecting points")
    eval_parser.add_argument(
        "--evaluator-cls",
        type=ClassType(EvaluatorBase),
        metavar="evaluator",
        default=Sequential,
    )
    eval_parser.add_argument(
        "--evaluator-params",
        type=KeyValuePairsType(),
        metavar="key=value,[...]",
        default=dict(),
    )


def make_plot_parser(parser):
    """Add common plotting options."""
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument("--save-file", type=OutputFileType())
    mode_parser.add_argument(
        "--interactive",
        help="launch in interactive mode (mutually exclusive with --save-file)",
        action="store_true",
    )

    mw.add_parser_config_args(parser)

    parser.add_argument("--res-dir", type=str, required=True)
    parser.add_argument(
        "--skip-pats",
        type=str,
        nargs="*",
        default=[],
        help="files matching these regular expressions are skipped",
    )
    parser.add_argument(
        "--log-scale", action="store_true", help="use log scale for y-axis"
    )


class TensorboardWriterType(OutputDirectoryType):
    def __call__(self, string):
        if string is None:
            return None
        super().__call__(string)
        tb_writer = SummaryWriter(string, purge_step=0)
        logging.info(f"created tensorboard writer for {string}")
        return tb_writer
