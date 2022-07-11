
import torch
import numpy as np
import utils.la1

def dot(a, b):
    result = a*b
    result = result.sum(-1, keepdims=True)
    return result

def normalize_sum(x):
    mean_val = x.mean(-1, keepdims=True)
    x = x/mean_val

    return x

def reflect(x):
    a = x[...,:2]
    b = x[...,2:]
    return torch.cat([-a, b],dim=-1)



def reflect_over_normal(d,n):
    r = d - 2* utils.la1.dot(d,n)*n
    return r




def get_rotation_matrix_params(x):
    theta = torch.tanh(x[...,0])*np.pi/2.0
    u_x = torch.tanh(x[...,1])
    u_y = torch.tanh(x[:,2])
    u_z = torch.sqrt((1 - u_x*u_x - u_y*u_y).clamp(min=0))

    return theta, u_x, u_y, u_z


# [N, 4] -> [N, 3, 3]
def get_rotation_matrix(x):
    theta, u_x, u_y, u_z = get_rotation_matrix_params(x)

    cos_theta = torch.cos(theta)
    one_cos_theta = 1 - cos_theta
    sin_theta = torch.sin(theta)

    a_11 = cos_theta +  u_x * u_x * one_cos_theta
    a_12 = u_x * u_y * one_cos_theta - u_z * sin_theta
    a_13 = u_x * u_z * one_cos_theta + u_y * sin_theta

    a_21 = u_y * u_x * one_cos_theta + u_z * sin_theta
    a_22 = cos_theta + u_y * u_y * one_cos_theta
    a_23 = u_y * u_z * one_cos_theta - u_x * sin_theta

    a_31 = u_z * u_x * one_cos_theta - u_y * sin_theta
    a_32 = u_z * u_y * one_cos_theta  + u_x * sin_theta
    a_33 = cos_theta + u_z*u_z*one_cos_theta

    a__1 = torch.stack([a_11, a_21, a_31], -1)
    a__2 = torch.stack([a_12, a_22, a_32], -1)
    a__3 = torch.stack([a_13, a_23, a_33], -1)


    a = torch.stack([a__1, a__2, a__3], -1) # is it correct?

    return a


def matrix_multiply(m: torch.Tensor, v):

    v = v.unsqueeze(-1)

    result = torch.matmul(m, v)

    result = result[...,0]

    return result












