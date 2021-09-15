"""models_bpnn.py: models and acquisitions for BPNN using TF2 and TensorMol."""

import tensorflow as tf
from typing import Iterable

from infopt.ihvp_tf import LowRankIHVPTF
from infopt.nnmodel_tf import NNModelTF
from infopt.nnmodel_mcd_tf import NNModelMCDTF
from infopt.nnacq_tf import NNAcqCategoricalTF


# import TensorMol as tm


DEVICE = "gpu"  # placeholder


def make_bpnn(
        input_dim: int = 8,
        layer_sizes: Iterable[int] = (26, 26),
        activation: str = "tanh",
        dropout: float = 0.0,
        dtype: tf.dtypes.DType = tf.float64,
):
    """Construct an atomic BPNN for energy calculation.

    Molecular energy can be computed by feeding each molecule as a batch and
    summing over their atomic energies.

    This is a simplified version that uses the same NN for all atoms.
    """
    # Input: [num_mols, (num_atoms), input_dim]
    inputs = tf.keras.Input(shape=[None, input_dim], dtype=dtype, ragged=True)

    # workaround to tf2's issues with RaggedTensors: pad & mask
    masking_layer = tf.keras.layers.Masking(mask_value=0.0)
    x = masking_layer(inputs.to_tensor(0.0))
    for layer_size in layer_sizes:
        x = tf.keras.layers.Dense(layer_size, activation=activation)(x)
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(dropout)(x)
    # Output: sum over atomic energies
    x = tf.squeeze(tf.keras.layers.Dense(1)(x), -1)
    outputs = tf.reduce_sum(x, axis=-1, keepdims=True)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BPNN")
    return model


def model_bpnn_inf(base_model, space, args, feature_map=None):
    """Neural network model with influence acquisition using TF2.

    Returns (model, acq).
    """
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

    args.bom_optim_params["learning_rate"] = args.bom_optim_lr_scheduler_cls(
        args.bom_optim_params["learning_rate"],
        **args.bom_optim_lr_scheduler_params,
    )
    bo_model_optim = args.bom_optim_cls(**args.bom_optim_params)
    bo_model = NNModelTF(
        base_model,
        ihvp,
        bo_model_optim,
        args.bom_loss_cls(reduction=tf.keras.losses.Reduction.NONE),
        args.bom_up_batch_size,
        args.bom_up_iters_per_point,
        args.bom_up_upsample_new,
        args.bom_n_higs,
        args.bom_ihvp_n,
        args.bom_weight_decay,
        args.bom_ckpt_every,
        DEVICE,
        args.tb_writer,
        feature_map=feature_map,
    )

    acq = NNAcqCategoricalTF(
        bo_model,
        space,
        0,  # set in optimization.py#L88 (using args)
        args.acq_n_candidates,
        args.acq_batch_size,
        args.tb_writer,
        args.acq_reinit_optim_start,
        None,  # use random sampler
        feature_map,
    )

    return bo_model, acq


def model_bpnn_mcd(base_model, space, args, feature_map=None):
    """Neural network model with MC Dropout uncertainty-based acquisition
    using TF2.

    Returns (model, acq).
    """
    args.bom_optim_params["learning_rate"] = args.bom_optim_lr_scheduler_cls(
        args.bom_optim_params["learning_rate"],
        **args.bom_optim_lr_scheduler_params,
    )
    bo_model_optim = args.bom_optim_cls(**args.bom_optim_params)
    bo_model = NNModelMCDTF(
        base_model,
        bo_model_optim,
        args.bom_loss_cls(reduction=tf.keras.losses.Reduction.NONE),
        args.bom_up_batch_size,
        args.bom_up_iters_per_point,
        args.bom_up_upsample_new,
        args.mcd_dropout,
        args.mcd_n_dropout_samples,
        args.mcd_lengthscale,
        args.mcd_tau,
        args.bom_ckpt_every,
        DEVICE,
        args.tb_writer,
        feature_map=feature_map,
    )

    acq = NNAcqCategoricalTF(
        bo_model,
        space,
        0,  # set in optimization.py#L88 (using args)
        args.acq_n_candidates,
        args.acq_batch_size,
        args.tb_writer,
        args.acq_reinit_optim_start,
        None,  # use random sampler
        feature_map,
    )

    return bo_model, acq
