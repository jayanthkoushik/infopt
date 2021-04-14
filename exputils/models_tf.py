"""models_tf.py: TF2 models for experiments."""

import tensorflow as tf
from typing import List


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

