
import os

import subprocess

def main():
    import torch

    path_base = os.path.dirname(os.path.abspath(__file__))

    path_cpputils_build = os.path.join(path_base, "cpputils/build")


    os.makedirs(path_cpputils_build, exist_ok=True)
    os.chdir(path_cpputils_build)


    torch_path = os.path.dirname(os.path.abspath(torch.__file__))
    torch_path_cmake_dir = os.path.join(torch_path, "share/cmake/Torch")

    torch_cuda_ver = torch.version.cuda

    cuda_path = "/usr/local/cuda-" + torch_cuda_ver


    print(torch.__file__)

    cmd = ["cmake", '-D', 'Torch_DIR='+torch_path_cmake_dir, "-DCMAKE_BUILD_TYPE=Release",
           '-D', "CUDA_TOOLKIT_ROOT_DIR="+cuda_path,
           ".."]
    print(cmd)
    subprocess.run(cmd)
    subprocess.run(["cmake", "--build", ".", "-j", "8"])

    pass


if __name__ == "__main__":
    main()