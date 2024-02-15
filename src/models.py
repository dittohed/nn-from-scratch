import numpy as np

from utils import append_ones


class MLP:
    """
    Implementation of basic multilayer perceptron (consisting of
    linear layers and activations only).
    """

    def __init__(self, layers: list):
        self.layers = layers

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

        for layer in self.layers:
            logits = x @ layer.w
            x = append_ones(layer.activation_fn(logits))

            if store_grad:
                layer.activations = x
                layer.d_activations = layer.activation_fn.d_fn(logits)

        return x[:, :-1]  # Discard appended ones

    def backward(self, x: np.array, lr: float, momentum: float = 0.9) -> None:
        """
        Perform full backward pass.

        Args:
            x (np.array): 
                Input data.
            lr (float):
                Learning rate applied to all layers.
            momentum (float):
                Momentum factor for SGD.
        """

        batch_size = x.shape[0]

        for i, layer in reversed(list(enumerate(self.layers))):
            if i > 0:
                prev_layer = self.layers[i-1]
                prev_layer.delta = (layer.delta @ layer.w[:-1].T) * prev_layer.d_activations
                prev_activations = prev_layer.activations
            else:
                prev_activations = x

            # Take average gradient (not total)
            grad = (prev_activations.T @ layer.delta) / batch_size

            layer.m = momentum * layer.m - lr * grad
            layer.w += layer.m