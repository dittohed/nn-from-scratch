import numpy as np

from tqdm import tqdm

from losses import Loss
from utils import append_ones


class MLP:
    """
    TODO
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        for layer in self.layers:
            logits = x @ layer.w
            x = append_ones(layer.activation_fn(logits))

            if store_grad:
                layer.activations = x
                layer.d_activations = layer.activation_fn.d_fn(logits)

        return x[:-1]  # Ignore appended ones

    def backward(self, x: np.array, lr: float) -> None:
        batch_size = x.shape[0]

        for i, layer in reversed(list(enumerate(self.layers))):
            if i > 0:
                prev_layer = self.layers[i-1]
                prev_layer.delta = (layer.w[:-1] @ layer.delta.T) * prev_layer.d_activations.T

                prev_activations = prev_layer.activations
            else:
                prev_activations = x

            # Take average, not total gradient
            grad = (prev_activations.T @ layer.delta) / batch_size 
            layer.w -= lr * grad

    def train(self, data: list, loss: Loss, lr: float) -> None:
        for (x, y_true) in tqdm(data):
            y_proba = self.forward(x)
            self.layers[-1].delta = loss.delta_fn(y_true, y_proba)
            self.backward(x, lr)
