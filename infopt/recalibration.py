"""
Uncertainty recalibrators

References:
    Kuleshov et al. (2018), Chung et al. (2021)
    https://github.com/uncertainty-toolbox/uncertainty-toolbox

Installation:
    poetry add git+https://github.com/uncertainty-toolbox/uncertainty-toolbox.git
"""

from typing import Callable
import numpy as np

import uncertainty_toolbox as uct


def get_recalibrator(
        y_pred: np.ndarray,
        y_std: np.ndarray,
        y_true: np.ndarray,
        mode: str = "interval",
        coverage: float = 0.95,
        custom_scalar: float = 1.,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Obtain a recalibrator using the uncertainty-toolbox package.

    Expected use case is when you try to obtain a simple recalibrator
    for a trained neural network on a separate calibration set.

    Supported modes:
        - std (optimized scalar)
        - interval (gaussian quantile/interval at a coverage level, using isotonic regression)
        - custom (user-input scalar)
    """

    if mode not in ["std", "interval", "custom"]:
        raise ValueError(f"unrecognized recalibrator name {mode}")

    def recalibrator(new_pred, new_std):
        if mode == "std":
            std_recalibrator = uct.recalibration.get_std_recalibrator(
                y_pred, y_std, y_true,
            )
            return std_recalibrator(new_std)

        elif mode == "interval":
            interval_recalibrator = uct.recalibration.get_interval_recalibrator(
                y_pred, y_std, y_true,
            )
            interval = interval_recalibrator(new_pred, new_std, coverage)
            return interval.upper - interval.lower

        # custom
        else:
            return custom_scalar * new_std

    return recalibrator
