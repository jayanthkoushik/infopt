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


# def TFSymSet_Linear_WithEle_Release(
#         R, Zs,
#         eles_, SFPsR_, Rr_cut, eleps_, SFPsA_, zeta, eta, Ra_cut,
#         RadpEle, AngtEle, mil_j, mil_jk,
# ):
#     import TensorMol as tm
#     inp_shp = tf.shape(R)
#     nmol = inp_shp[0]
#     natom = inp_shp[1]
#     GMR = tf.reshape(
#         tm.TFSymRSet_Linear_WithEle_Release(
#             R, Zs, eles_, SFPsR_, eta, Rr_cut, RadpEle, mil_j
#         ), [nmol, natom,-1])
#     GMA = tf.reshape(
#         tm.TFSymASet_Linear_WithEle(
#             R, Zs, eleps_, SFPsA_, zeta,  eta, Ra_cut,  AngtEle, mil_jk
#         ), [nmol, natom,-1])
#     return tf.concat([GMR, GMA], axis=2)

def make_bpnn_tm(params: dict):
    """Construct the BPNN model using TensorMol."""

    import TensorMol as tm

    tf_prec = tf.float64
    tf_int = tf.int64

    # Variables: comes from TensorMolData.GetTrainBatch()
    tmdata = tm.TensorMolData()
    instance = tm.MolInstance_DirectBP_EandG_SymFunction(tmdata)

    # inputs
    batch_size = 16
    xyzs, Zs, Elabels, grads, Radp_Ele, Angt_Elep, mil_j, mil_jk, natom_inv = \
        tmdata.GetTrainBatch(batch_size)

    xyzs = tf.convert_to_tensor(xyzs, dtype=tf_prec)
    Zs = tf.convert_to_tensor(Zs, dtype=tf_int)
    Elabels = tf.convert_to_tensor(Elabels, dtype=tf_prec)
    grads = tf.convert_to_tensor(grads, dtype=tf_prec)
    Radp_Ele = tf.convert_to_tensor(Radp_Ele, dtype=tf_int)
    Angt_Elep = tf.convert_to_tensor(Angt_Elep, dtype=tf_int)
    mil_j = tf.convert_to_tensor(mil_j, dtype=tf_int)
    mil_jk = tf.convert_to_tensor(mil_jk, dtype=tf_int)
    natom_inv = tf.convert_to_tensor(natom_inv, dtype=tf_prec)

    # Constants: Comes from MolInstance*.__init__()
    Ele = tf.convert_to_tensor(instance.eles_np, dtype=tf_int)
    Elep = tf.convert_to_tensor(instance.eles_pairs_np, dtype=tf_int)
    # (SetANI1Param within __init__)
    Ra_cut = tf.convert_to_tensor(instance.Ra_cut, dtype=tf_prec)
    Rr_cut = tf.convert_to_tensor(instance.Rr_cut, dtype=tf_prec)
    SFPr2 = tf.convert_to_tensor(instance.SFPr2, dtype=tf_prec)
    SFPa2 = tf.convert_to_tensor(instance.SFPa2, dtype=tf_prec)
    zeta = tf.convert_to_tensor(instance.zeta, dtype=tf_prec)
    eta = tf.convert_to_tensor(instance.eta, dtype=tf_prec)

    # params
    keep_probs = tm.PARAMS["KeepProb"]  # layer-by-layer

    # Preprocessing
    scatter_sym, sym_index = tm.TFSymSet_Scattered_Linear_WithEle_Release(
        xyzs, Zs,
        Ele, SFPr2, Rr_cut, Elep, SFPa2, zeta, eta, Ra_cut,
        Radp_Ele, Angt_Elep, mil_j, mil_jk
    ) # List[shape(*, 1120)], List[shape(*, 2)]
    # The BPNN network
    Etotal, Ebp, Ebp_atom = instance.energy_inference(
        scatter_sym, sym_index, xyzs, keep_probs,
    )
    # Loss calculation (involves the gradient)
    gradient = tf.gradients(Etotal, xyzs, name="BPEGrad")
    total_loss, loss, energy_loss, grads_loss = instance.loss_op(
        Etotal, gradient, Elabels, grads, 1 / natom_inv
    )

    return total_loss, loss, energy_loss, grads_loss


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
