import numpy as np

from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Base class for activation functions.

    At the moment, the objects of this class don't calculate the loss
    explicitly as it's not needed for performing backpropagation.
    """

    # TODO: is `y_proba` used in all the losses?
    @abstractmethod
    def delta_fn(self, y_proba: np.array, y_true: np.array) -> np.array:
        """
        Calculate derivative of the loss function
        with respect to logits.

        Args:
            y_proba (np.array): 
                Predicted probabilities.
            y_true (np.array):
                True probabilities.
        """

        pass


class MultiCrossEntropy(Loss):
    """
    Implementation of multi-class cross-entropy.

    Assumes that true labels are one-hot encoded.
    """

    def delta_fn(self, y_proba: np.array, y_true: np.array) -> np.array:
        return y_proba - y_true