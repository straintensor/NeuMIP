//
// Created by alex on 2/23/21.
//

#ifndef CPPACC_GPU_TORCH_UTILS_H
#define CPPACC_GPU_TORCH_UTILS_H

#include <torch/extension.h>

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define CHECK_TENSOR_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_TENSOR_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR_INPUT(x) CHECK_TENSOR_CUDA(x); CHECK_TENSOR_CONTIGUOUS(x)

typedef float TFloat;
typedef torch::PackedTensorAccessor64<TFloat,4,torch::RestrictPtrTraits> AccFloat4;
typedef torch::PackedTensorAccessor64<TFloat,3,torch::RestrictPtrTraits> AccFloat3;


class JobSize{
public:
    int batch, height, width;

    __host__ __device__ bool calc_position(JobSize &cur_loc, size_t size) {
        cur_loc.width = size % width;
        size /= width;
        cur_loc.height = size % height;
        size /= height;
        cur_loc.batch = size % batch;
        size /= batch;
        return size == 0;

    }

    size_t get_length() {
        return batch*height*width;
    }

    size_t get_num_blocks(size_t block_size) {
        return (get_length()-1)/block_size + 1;
    }


};


#endif //CPPACC_GPU_TORCH_UTILS_H
