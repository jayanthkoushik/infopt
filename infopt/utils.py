"""
TF2 tensor slicing/indexing utility functions
"""

from typing import List, Tuple, Union
import numpy as np
import tensorflow as tf


def normalize_output(
        out: Union[tf.Tensor, Tuple[tf.Tensor], List[tf.Tensor]],
        index: int = 0,
) -> tf.Tensor:
    """Normalize output of a `tf.keras.Model` into a `tf.Tensor`,
    especially when the model outputs a tuple."""
    if isinstance(out, tuple) or isinstance(out, list):
        return out[index]
    return out


def slice_tensors(
        tensors: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]],
        start: int = None,
        end: int = None,
):
    """Slice over the first dimension of the input tensor(s) to
    a `tf.keras.Model`."""
    if isinstance(tensors, tuple):
        return tuple(t[start:end] for t in tensors)
    elif isinstance(tensors, list):
        return list(t[start:end] for t in tensors)
    return tensors[start:end]


def gather_tensors(
        tensors: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]],
        indices: Union[List[int], np.ndarray, tf.Tensor],
        axis: int = 0,
):
    """Gather over a dimension of the input tensor(s)."""
    if isinstance(tensors, tuple):
        return tuple(tf.gather(t, indices, axis=axis) for t in tensors)
    elif isinstance(tensors, list):
        return list(tf.gather(t, indices, axis=axis) for t in tensors)
    return tf.gather(tensors, indices, axis=axis)


def get_size(
    tensors: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]]
) -> int:
    """Get size of the first dimension of the tensor(s)."""
    if isinstance(tensors, tuple) or isinstance(tensors, list):
        return tensors[0].shape[0] if len(tensors) >= 1 else None
    return tensors.shape[0]
