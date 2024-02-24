from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Base class for neural network layers.

    The objects of this class can store intermediate values
    for backpropagation.
    """
    
    def __init__(self):
        self._x = None  # For storing input during `forward`

    @abstractmethod
    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        """
        Call the layer on input.

        Args:
            x (np.array): 
                Input.
            store_grad (bool): 
                Whether to store input for further gradient calculation.

        Returns:
            np.array: Layer output.
        """
        pass


class LayerTrainable(Layer):
    """
    Base class for trainable neural network layers.
    """

    @abstractmethod
    def backward(self, delta: np.array, lr: float, momentum: float) -> np.array:
        """
        Calculate gradients with respect to the layer's parameters 
        based on gradients with respect to the layer's output. 
        
        Update the layer's parameters.
        
        Return gradients with respect to the layer's input.

        Args:
            delta (np.array):
                Gradients with respect to the layer's output.
            lr (float):
                Learning rate.
            momentum (float):
                Momentum factor for SGD.

        Returns:
            np.array: Gradients with respect to the layer's input.
        """
        pass


    @abstractmethod
    def reset(self) -> None:
        """
        Reset layer's params.
        """
        pass


class LayerNonTrainable(Layer):
    """
    Base class for non-trainable neural network layers.
    """

    @abstractmethod
    def backward(self, delta: np.array) -> np.array:
        """
        Calculate gradients with respect to the layer's parameters 
        based on gradients with respect to the layer's output. 

        Args:
            delta (np.array):
                Gradients with respect to the layer's output.
 
        Returns:
            np.array: Gradients with respect to the layer's input.
        """
        pass


class Linear(LayerTrainable):
    """
    Implementation of basic linear layer. 
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

        self._w = None
        self._b = None

        self._mw = None
        self._mb = None

        self.reset()

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        if store_grad:
            self._x = x
        
        return x @ self._w + self._b
    
    def backward(self, delta: np.array, lr: float, momentum: float) -> np.array:
        batch_size = delta.shape[0]

        # Take average gradient (not total)
        grad_w = (self._x.T @ delta) / batch_size
        grad_b = np.sum(delta, axis=0) / batch_size

        self._mw = momentum * self._mw - lr * grad_w
        self._mb = momentum * self._mb - lr * grad_b

        self._w += self._mw
        self._b += self._mb

        delta_prev = delta @ self._w.T
        return delta_prev

    def reset(self) -> None:
        # Init weights as in 
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        limit = np.sqrt(1/self._input_size)
        self._w = np.random.uniform(
            -limit, limit, size=(self._input_size, self._output_size)
        ).astype(np.float32)
        self._b = np.random.uniform(
            -limit, limit, size=(1, self._output_size)
        ).astype(np.float32)

        # Momentum
        self._mw = np.zeros_like(self._w)
        self._mb = np.zeros_like(self._b)


class Sigmoid(LayerNonTrainable):
    """
    Implementation of sigmoid function.
    """

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        if store_grad:
            self._x = x

        return self._sigmoid(x)

    def backward(self, delta: np.array) -> np.array:
        x = self._sigmoid(self._x)
        return delta * x * (1 - x)
    
    def _sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))


class Softmax(LayerNonTrainable):
    """
    Implementation of softmax function.
    """

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        # Normalize with max logit to avoid NaNs 
        # (doesn't affect the output values)
        x -= np.max(x, axis=-1, keepdims=True)
        x = np.exp(x)
        denom = np.sum(x, axis=-1, keepdims=True)
        x = x / denom

        return x
    
    def backward(self, delta: np.array) -> np.array:
        raise NotImplementedError('Calling .backward() not supported for objects of `Softmax`.')