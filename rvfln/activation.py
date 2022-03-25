from typing import Callable

import numpy as np


ActivationFunction = Callable[[np.ndarray], np.ndarray]


def ReLU(a: np.ndarray) -> np.ndarray:
    """
    Simple numpy implementation of Rectified Linear Unit
    """
    return np.maximum(a, 0)


def LeakyReLU(a: np.ndarray) -> np.ndarray:
    """
    Simple numpy implementation of LeakyReLU
    """
    return np.where(a > 0, a, a * 0.01)
