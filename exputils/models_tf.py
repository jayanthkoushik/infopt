"""models_tf.py: TF2 models for experiments."""

import tensorflow as tf
from typing import List

from infopt.ihvp_tf import LowRankIHVPTF
from infopt.nnmodel_tf import NNModelTF
from infopt.nnacq_tf import NNAcqTF


# TODO(yj): device control for TF2
DEVICE = "cpu"


def make_fcnet(
        in_dim: int,
        layer_sizes: List[int],
        nonlin: str = "relu",
        dropout: float = 0.0,
):
    """Construct a tf.keras FC network for regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(in_dim, ))
    ])
    for layer_size in layer_sizes:
        model.add(tf.keras.layers.Dense(layer_size, activation=nonlin))
        if dropout > 0.0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    return model


def model_nn_inf(base_model, space, args, acq_fast_project=None):
    """Neural network model with influence acquisition."""
    # Returns (model, acq).
    ihvp = LowRankIHVPTF(
        base_model.trainable_variables,
        args.ihvp_rank,
        args.ihvp_batch_size,
        args.ihvp_iters_per_point,
        args.ihvp_loss_cls(),
        args.ihvp_optim_cls,
        args.ihvp_optim_params,
        args.ihvp_ckpt_every,
        DEVICE,
        args.tb_writer,
    )

    bo_model_optim = args.bom_optim_cls(**args.bom_optim_params)
    bo_model = NNModelTF(
        base_model,
        ihvp,
        bo_model_optim,
        args.bom_loss_cls(reduction=tf.keras.losses.Reduction.NONE),
        args.bom_up_batch_size,
        args.bom_up_iters_per_point,
        args.bom_n_higs,
        args.bom_ihvp_n,
        args.bom_ckpt_every,
        DEVICE,
        args.tb_writer,
    )

    acq = NNAcqTF(
        bo_model,
        space,
        0,
        args.acq_optim_cls,
        args.acq_optim_params,
        args.acq_optim_iters,
        args.acq_ckpt_every,
        acq_fast_project,
        args.acq_rel_tol,
        DEVICE,
        args.tb_writer,
        args.acq_reinit_optim_start,
        args.acq_optim_lr_decay_step_size,
        args.acq_optim_lr_decay_gamma,
    )

    return bo_model, acq
