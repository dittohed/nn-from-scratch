import numpy as np


class Activation:
    def __call__(self, x: np.array) -> np.array:
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
        x -= np.max(x, axis=-1, keepdims=True)
        x = np.exp(x)
        denom = np.sum(x, axis=-1, keepdims=True)
        x = x / denom

        return x
    
    def d_fn(self, x: np.array) -> np.array:
        pass