import numpy as np


def append_ones(x: np.array):
    return np.hstack(
        (x, np.ones((x.shape[0], 1), dtype=np.float32))
    )