import numpy as np


def one_hot(dense, ndim=10):
    N = dense.shape[0]
    ret = np.zeros([N, ndim])
    ret[np.arange(N), dense] = 1
    return ret

