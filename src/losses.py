import numpy as np


class Loss:
    def __call__(self, y_proba: np.array, y_true: np.array) -> np.array:
        pass

    def delta_fn(self, y_proba: np.array, y_true: np.array) -> np.array:
        pass


# TODO: add warning that this works for one-hot `y_true` only`
class MultiCrossEntropy(Loss):
    def __call__(self, y_proba: np.array, y_true: np.array) -> np.array:
        pass

    def delta_fn(self, y_proba: np.array, y_true: np.array) -> np.array:
        return y_proba - y_true