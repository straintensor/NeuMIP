


def add_build_path():
    import os
    import sys

    path_base = os.path.dirname(os.path.abspath(__file__))
    cppacc_path = os.path.join(path_base, "build")


    sys.path.append(cppacc_path)



try:
    import importlib
    if importlib.util.find_spec("cppacc") is None:
        add_build_path()


    import cppacc

    TorchActiveModel = cppacc.TorchActiveModel

except ModuleNotFoundError as e:
    add_build_path()


    print("Skipping importing cpputils")

import torch


def texture_interpolate(uv_tensor: torch.Tensor, texture_tensor_0: torch.Tensor):
    uv_tensor = uv_tensor.contiguous()
    texture_tensor_0 = texture_tensor_0.contiguous()

    assert len(texture_tensor_0.shape) == 3
    assert len(uv_tensor.shape) == 4
    assert uv_tensor.shape[1] == 2

    assert uv_tensor.type() ==  'torch.cuda.FloatTensor'
    assert texture_tensor_0.type() ==  'torch.cuda.FloatTensor'

    return cppacc.texture_interpolate(uv_tensor, texture_tensor_0)


