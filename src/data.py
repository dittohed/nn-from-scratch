import numpy as np


class NumpyDataLoader:
    """
    Simplified custom implementation of `torch.utils.data.DataLoader` that
    operates on NumPy arrays.
    """

    def __init__(
        self, x: np.array, y: np.array, batch_size: int, shuffle: bool = False
    ):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._shuffle = shuffle
        
        self._reset()

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self._x)

    def __next__(self) -> tuple:
        if self._index < len(self._x):
            data = self._x[self._index], self._y[self._index]
            self._index += 1
            return data
        else:
            self._reset(from_batches=True)
            raise StopIteration
        
    def _reset(self, from_batches=False) -> None:
        self._index = 0

        if from_batches:
            self._x = np.concatenate(self._x)
            self._y = np.concatenate(self._y)

        if self._shuffle:
            random_order = np.random.permutation(len(self._x))
            self._x = self._x[random_order]
            self._y = self._y[random_order]

        self._x = self._split_into_batches(self._x)
        self._y = self._split_into_batches(self._y)

    def _split_into_batches(self, x: np.array) -> np.array:
        x = [
            x[pos: min(pos + self._batch_size, len(x))]
            for pos in range(0, len(x), self._batch_size)
        ]

        return x