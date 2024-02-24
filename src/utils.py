import numpy as np


def preprocess_mnist_imgs(x: np.array, img_size: int):
    x = x.astype(np.float32) / 255
    x = x.reshape(-1, img_size*img_size)

    return x


def onehot(y: np.array):
    y_onehot = np.zeros(
        (len(y), np.max(y)+1), dtype=np.float32
    ) 
    y_onehot[np.arange(len(y)), y] = 1

    return y_onehot


def accuracy(y_true: np.array, y_pred: np.array):
    return np.sum(y_true == y_pred) / len(y_true)