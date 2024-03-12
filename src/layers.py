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
        Calculate and return gradients with respect to the layer's input 
        based on gradients with respect to the layer's output. 

        Calculate gradients with respect to the layer's parameters 
        based on gradients with respect to the layer's output and 
        update the layer's parameters.

        Args:
            delta (np.array):
                Gradients with respect to the layer's output.
            lr (float):
                Learning rate.
            momentum (float):
                Momentum factor for SGD.

        Returns:
            np.array: Gradients with respect to the layer's input 
                in the shape of input.
        """
        pass


    @abstractmethod
    def reset(self) -> None:
        """
        Reset layer's params and associated variables.
        """
        pass


class LayerNonTrainable(Layer):
    """
    Base class for non-trainable neural network layers.
    """

    @abstractmethod
    def backward(self, delta: np.array) -> np.array:
        """
        Calculate and return gradients with respect to the layer's input 
        based on gradients with respect to the layer's output. 

        Args:
            delta (np.array):
                Gradients with respect to the layer's output.
 
        Returns:
            np.array: Gradients with respect to the layer's input
                in the shape of input.
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
        delta_prev = delta @ self._w.T

        grad_w = (self._x.T @ delta) / batch_size
        grad_b = np.sum(delta, axis=0) / batch_size

        self._mw = momentum * self._mw - lr * grad_w
        self._mb = momentum * self._mb - lr * grad_b

        self._w += self._mw
        self._b += self._mb

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


class Conv2d(LayerTrainable):
    """
    Implementation of convolution with 2D kernels.
    """

    def __init__(
        self, in_channels: int, out_channels: int, k: int, stride: int = 1, 
        padding: int = 0
    ):
        super().__init__()

        self._k = k  # Kernel size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._padding = padding

        self._w = None
        self._b = None
        self._mw = None
        self._mb = None
        self.reset()

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        self._determine_output_shape(x.shape)

        x = self._pad(x, self._padding, self._padding)
        self._in_padded_shape = x.shape

        x = self._im2col(x)
        if store_grad:
            self._x = x

        x = self._w @ x + self._b  # Actual convolution

        x = x.reshape(
            self._out_channels, self._batch_size, self._h_out, self._w_out
        ) 
        x = x.transpose(1, 0, 2, 3)
    
        return x
    
    def backward(self, delta: np.array, lr: float, momentum: float) -> np.array:
        # Reshape gradients back to _im2col() result shape
        delta = delta.transpose(1, 0, 2, 3)
        delta = delta.reshape(self._out_channels, -1)
        delta_prev = self._w.T @ delta

        # At this point gradient was calculated only with respect to result
        # of im2col, not actual input
        delta_prev = self._col2im(delta_prev)

        # Padding wasn't output by the previous layer, so it should be removed
        delta_prev = self._unpad(delta_prev, self._padding, self._padding)

        grad_w = (delta @ self._x.T) / self._batch_size
        grad_b = np.sum(delta, axis=1) / self._batch_size

        # Add dummy dim to avoid broadcasting below
        grad_b = np.expand_dims(grad_b, axis=1)

        self._mw = momentum * self._mw - lr * grad_w
        self._mb = momentum * self._mb - lr * grad_b

        self._w += self._mw
        self._b += self._mb

        return delta_prev

    def reset(self) -> None:
        # Init weights as in 
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        limit = np.sqrt(
            1 / (self._in_channels * self._k * self._k)
        )
        self._w = np.random.uniform(
            -limit, limit, size=(self._out_channels, self._k * self._k * self._in_channels)
        ).astype(np.float32)
        self._b = np.random.uniform(
            -limit, limit, size=(self._out_channels, 1)
        ).astype(np.float32)

        # Momentum
        self._mw = np.zeros_like(self._w)
        self._mb = np.zeros_like(self._b)

        self._cached_indices = None
        self._batch_size = None
        self._in_padded_shape = None
        self._h_out = None
        self._w_out = None
        self._shape_changed = True

    def _determine_output_shape(self, x_shape: tuple) -> None:
        """
        Determine output shape. This is needed in case batch size
        or spatial dims of input change at some point.
        """

        batch_size, _, h_in, w_in = x_shape
        h_out = int((h_in + 2 * self._padding - self._k) / self._stride) + 1
        w_out = int((w_in + 2 * self._padding - self._k) / self._stride) + 1

        if (batch_size != self._batch_size or h_out != self._h_out
                or w_out != self._w_out):
            self._shape_changed = True  # im2col indices will be recalculated
            self._batch_size = batch_size
            self._h_out = h_out
            self._w_out = w_out
        else:
            self._shape_changed = False
    
    def _im2col(self, x: np.array) -> np.array:
        """
        Transform an array of shape `[B, C, H, W]` into columns, where
        each column corresponds to a flattened receptive field (including 
        all input channels). If `B` > 1, just more columns are created (no batch
        dim is retained).

        Example:
        >>> x = np.arange(32).reshape(2, 1, 4, 4)  # Two 1D 4x4 images
        >>> x
        array([[[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11],
                 [12, 13, 14, 15]]],

               [[[16, 17, 18, 19],
                 [20, 21, 22, 23],
                 [24, 25, 26, 27],
                 [28, 29, 30, 31]]]])
        >>> conv = Conv2d(1, 1)
        >>> conv.forward(x)  # Cache indices
        >>> conv._im2col(x)
        array([[ 0,  1,  4,  5, 16, 17, 20, 21],
               [ 1,  2,  5,  6, 17, 18, 21, 22],
               [ 2,  3,  6,  7, 18, 19, 22, 23],
               [ 4,  5,  8,  9, 20, 21, 24, 25],
               [ 5,  6,  9, 10, 21, 22, 25, 26],
               [ 6,  7, 10, 11, 22, 23, 26, 27],
               [ 8,  9, 12, 13, 24, 25, 28, 29],
               [ 9, 10, 13, 14, 25, 26, 29, 30],
               [10, 11, 14, 15, 26, 27, 30, 31]])

        Args:
            x (np.array):
                Array to be transformed.

        Returns:
            np.array: Columns with consecutive receptive fields.
        """

        if self._cached_indices is None or self._shape_changed:
            self._cached_indices = self._get_indices(x.shape, self._k, self._stride)

        i, j, d = self._cached_indices
        cols = x[:, d, i, j]  # Actual im2col operation
        cols = np.concatenate(cols, axis=-1)  # Flatten batch dimension
        
        return cols
    
    def _col2im(self, x: np.array) -> np.array:
        """
        Revert _im2col.

        If multiple elements of the _im2col output correspond to a single element
        of the _im2col input, they are added (to ensure correct gradient calculation).

        Example:
        >>> x = np.arange(32).reshape(2, 1, 4, 4)  # Two 1D 4x4 images
        >>> x
        array([[[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11],
                 [12, 13, 14, 15]]],

               [[[16, 17, 18, 19],
                 [20, 21, 22, 23],
                 [24, 25, 26, 27],
                 [28, 29, 30, 31]]]])
        >>> conv = Conv2d(1, 1)
        >>> conv.forward(x)  # Cache indices
        >>> conv._im2col(x)
        array([[ 0,  1,  4,  5, 16, 17, 20, 21],
               [ 1,  2,  5,  6, 17, 18, 21, 22],
               [ 2,  3,  6,  7, 18, 19, 22, 23],
               [ 4,  5,  8,  9, 20, 21, 24, 25],
               [ 5,  6,  9, 10, 21, 22, 25, 26],
               [ 6,  7, 10, 11, 22, 23, 26, 27],
               [ 8,  9, 12, 13, 24, 25, 28, 29],
               [ 9, 10, 13, 14, 25, 26, 29, 30],
               [10, 11, 14, 15, 26, 27, 30, 31]])
        >>> conv._col2im(self._im2col(x))
        array([[[[  0,   2,   4,   3],
                 [  8,  20,  24,  14],
                 [ 16,  36,  40,  22],
                 [ 12,  26,  28,  15]]],

               [[[ 16,  34,  36,  19],
                 [ 40,  84,  88,  46],
                 [ 48, 100, 104,  54],
                 [ 28,  58,  60,  31]]]])

        Args:
            x (np.array):
                Array to be transformed.

        Returns:
            np.array: Array restored from columns with consecutive receptive fields.
        """

        i, j, d = self._cached_indices
        x_in_shape = np.zeros(self._in_padded_shape)

        # Split columns into per-image columns: (C, N*B) => (B, C, N)
        x = np.array(np.hsplit(x, self._batch_size))

        # Reshape columns back to input shape
        # Overlapping elements will be added thanks to `at`
        np.add.at(x_in_shape, (slice(None), d, i, j), x)

        return x_in_shape
  
    @staticmethod
    def _pad(x: np.array, pad_h: int, pad_w: int) -> np.array:
        """
        Pad spatial dims (height and width) of an array of
        shape `[B, C, H, W]` with 0s.

        Args:
            x (np.array):
                Array to be padded.
            pad_h (int):
                How many 0s to add at the top and bottom.
            pad_w (int):
                How many 0s to add on the left and right.

        Returns:
            np.array: Padded array.
        """

        pad_dims = (
            (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)
        )
        return np.pad(x, pad_dims)

    @staticmethod
    def _unpad(x: np.array, pad_h: int, pad_w: int) -> np.array:
        """
        Unpad spatial dims (height and width) of an array of
        shape `[B, C, H, W]`.

        Args:
            x (np.array):
                Array to be unpadded.
            pad_h (int):
                How many elements to remove at the top and bottom.
            pad_w (int):
                How many elements to remove on the left and right.

        Returns:
            np.array: Unpadded array.
        """

        x = x[:, :, pad_h:-pad_h, :] if pad_h > 0 else x
        x = x[:, :, :, pad_w:-pad_w] if pad_w > 0 else x

        return x

    @staticmethod
    def _get_indices(img_shape: tuple, k: int, stride: int):
        """
        Calculate indices needed to transform an array of shape `[B, C, H, W]` 
        into columns, where each column corresponds to a flattened receptive field 
        (including all input channels).
        """

        _, c_in, h_in, w_in = img_shape

        out_h = int((h_in - k) / stride) + 1
        out_w = int((w_in - k) / stride) + 1

        # Upper-left corner index vector for i dimension (single channel)
        i = np.repeat(np.arange(k), k)
        # Duplicate for the other channels
        i = np.tile(i, c_in)
        # Prepare shifts to create index vector for all positions
        shifts_i = stride * np.repeat(np.arange(out_h), out_w)
        # Index vector for all positions (in columns)
        i = i.reshape(-1, 1) + shifts_i.reshape(1, -1)
        
        # Upper-left corner index vector for j dimension (single channel)
        j = np.tile(np.arange(k), k)
        j = np.tile(j, c_in)
        shifts_j = stride * np.tile(np.arange(out_w), out_h)
        j = j.reshape(-1, 1) + shifts_j.reshape(1, -1)

        # Add channel indices
        d = np.repeat(np.arange(c_in), k*k).reshape(-1, 1)

        return i, j, d
        

class Flatten(LayerNonTrainable):
    """
    Implementation of layer that flattens input to vector.
    """

    def __init__(self):
        super().__init__()
        self._x_in_shape = None

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        self._x_in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta: np.array) -> np.array:
        return delta.reshape(self._x_in_shape)


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
    

class ReLU(LayerNonTrainable):
    """
    Implementation of rectified linear unit.
    """

    def forward(self, x: np.array, store_grad: bool = True) -> np.array:
        if store_grad:
            self._x = x
        
        return np.maximum(0, x)
    
    def backward(self, delta: np.array) -> np.array:
        return delta * (self._x > 0).astype(np.float32) 
