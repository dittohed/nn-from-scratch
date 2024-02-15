import numpy as np

from activations import Activation


class Linear:
    """
    Implementation of basic linear layer. 

    The objects of this class store intermediate values
    for performing backpropagation and training with momentum SGD.
    """

    def __init__(
        self, input_size: int, output_size: int,
        activation_fn: Activation
    ):
        self._input_size = input_size
        self._output_size = output_size
        self.activation_fn = activation_fn

        self.reset()

    def reset(self) -> None:
        self.activations = None
        self.d_activations = None

        # Init weights as in 
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        limit = np.sqrt(1/self._input_size)
        self.w = np.random.uniform(
            -limit, limit, size=(self._input_size+1, self._output_size)
        ).astype(np.float32)

        self.m = np.zeros_like(self.w)
