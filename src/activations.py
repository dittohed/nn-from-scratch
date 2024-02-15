import numpy as np

from abc import ABC, abstractmethod


class Activation(ABC):
    """
    Base class for activation functions.
    """

    @abstractmethod
    def __call__(self, x: np.array) -> np.array:
        """
        Call the activation function on input.

        Args:
            x (np.array): Input.
        """

        pass
    
    @abstractmethod
    def d_fn(self, x: np.array) -> np.array:
        """
        Calculate derivative of the activation function
        with respect to input.

        Args:
            x (np.array): Input.
        """

        pass


class Sigmoid(Activation):
    def __call__(self, x: np.array) -> np.array:
        return self._sigmoid(x)

    def d_fn(self, x: np.array) -> np.array:
        x = self._sigmoid(x)
        return x * (1 - x)
 
    def _sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))


class Softmax(Activation):
    def __call__(self, x: np.array) -> np.array:
        # Normalize with max logit to avoid NaNs 
        x -= np.max(x, axis=-1, keepdims=True)
        x = np.exp(x)
        denom = np.sum(x, axis=-1, keepdims=True)
        x = x / denom

        return x
    
    def d_fn(self, x: np.array) -> np.array:
        # Not used in the code
        pass