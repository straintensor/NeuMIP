
import torch

def get_3rd_axis(x):
    if x is None:
        return None

    x_a = x[:, 0:1, :, :]
    x_b = x[:, 1:2, :, :]

    result = torch.sqrt((1 - (x_a*x_a + x_b * x_b)).clamp(min=0))

    return result

def add_3rd_axis(x):
    assert x.shape[1] == 2
    x_3 = get_3rd_axis(x)
    return torch.cat([x, x_3], dim=1)

def add_3rd_axis_neg(x):
    assert x.shape[1] == 2
    x_3 = -get_3rd_axis(x)
    return torch.cat([x, x_3], dim=1)


def length(x: torch.Tensor):
    return torch.sqrt((x*x).sum(dim=1, keepdims=True))

def normalize(x: torch.Tensor, epsilon=1e-6):
    x_length = length(x) + epsilon
    return x/x_length


def vector_clamp(x, lim):
    x_length = length(x)
    x = x - x *torch.clamp(x_length - lim, min=0)
    return x

def dot(a, b):
    result = a*b
    result = result.sum(1, keepdims=True)
    return result

