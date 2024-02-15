import numpy as np


def append_ones(x: np.array):
    return np.hstack(
        (x, np.ones((x.shape[0], 1), dtype=np.float32))
    )


def preprocess_mnist_imgs(x: np.array, img_size: int):
    x = x.astype(np.float32) / 255
    x = x.reshape(-1, img_size*img_size)
    x = append_ones(x)

    return x


def onehot(y: np.array):
    y_onehot = np.zeros(
        (len(y), np.max(y)+1), dtype=np.float32
    ) 
    y_onehot[np.arange(len(y)), y] = 1

    return y_onehot


def accuracy(y_true: np.array, y_pred: np.array):
    return np.sum(y_true == y_pred) / len(y_true)