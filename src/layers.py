import numpy as np

from activations import Activation


class Linear:
    def __init__(
        self, input_size: int, output_size: int,
        activation_fn: Activation
    ):
        self._input_size = input_size
        self._output_size = output_size
        self._init_weights()

        self.activation_fn = activation_fn
        self.activations = None
        self.d_activations = None

    def _init_weights(self) -> None:
        limit = np.sqrt(1/self._input_size)
        self.w = np.random.uniform(
            -limit, limit, size=(self._input_size+1, self._output_size)
        ).astype(np.float32)

        self.m = np.zeros_like(self.w)
