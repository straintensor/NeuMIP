import numpy as np


def get_3rd_axis(x):
    if x is None:
        return None

    x_a = x[ :, :, 0:1]
    x_b = x[:, :, 1:2]

    result = np.sqrt(np.maximum(1 - (x_a*x_a + x_b * x_b), 0))

    return result

def get_circle(x):
    x_a = x[ :, :, 0:1]
    x_b = x[:, :, 1:2]

    return (x_a*x_a + x_b * x_b) < 1



def add_3rd_axis(x):

    assert x.shape[-1] == 2
    x_3 = get_3rd_axis(x)
    return np.concatenate([x, x_3], axis=-1)

def add_3rd_axis_neg(x):

    assert x.shape[-1] == 2
    x_3 = -get_3rd_axis(x)
    return np.concatenate([x, x_3], axis=-1)

def add_3rd_0(x: np.ndarray):
    assert x.shape[-1] == 2
    shape = list(x.shape)
    shape[-1] = 1
    b = np.zeros(shape, dtype=x.dtype)
    x = np.concatenate([x, b], axis=-1)
    return x

