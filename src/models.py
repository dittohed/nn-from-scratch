import numpy as np

from layers import LayerTrainable


class Model:
    """
    Implementation of neural network with a sequence of layers.
    """

    def __init__(self, layers: list):
        self._layers = layers

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        """
        Perform full forward pass.

        Args:
            x (np.array): 
                Input data.
            store_grad (bool):
                Whether to store intermediate values required to
                perform backward pass.
        """

        for layer in self._layers:
            x = layer.forward(x, store_grad)

        return x

    def backward(self, delta: np.array, lr: float, momentum: float = 0.9) -> None:
        """
        Perform full backward pass with momentum SGD.

        Args:
            delta (np.array): 
                Derivative of the loss function with respect to logits.
            lr (float):
                Learning rate applied to all layers.
            momentum (float):
                Momentum factor for SGD.
        """

        # Skip last activation layer as delta is already given
        # TODO: what if no activation at the end?
        for layer in reversed(self._layers[:-1]):
            if isinstance(layer, LayerTrainable):
                delta = layer.backward(delta, lr, momentum)
            else:
                delta = layer.backward(delta)