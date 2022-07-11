

#Format ..., channels

import torch

def cross(a, b):
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    return torch.stack([x,y,z], -1)


def length(a):
    a = a*a
    a = a.sum(-1, keepdims=True)
    return torch.sqrt(a)


def dot(a, b):
    result = a*b
    result = result.sum(-1, keepdims=True)
    return result


def normalize(a):
    a_len = length(a).clamp(min=1e-6)

    return a/a_len



def intersect_triangle(origin, direction, triangle):
    triangle_origin = triangle[..., :3]
    triangle_dir_b = triangle[..., 3:6]
    triangle_dir_c = triangle[..., 6:9]

    triangle_normal = cross(triangle_dir_b, triangle_dir_c)

    triangle_area = length(triangle_normal)



    time = dot(triangle_origin - origin, triangle_normal)/dot(direction, triangle_normal)






