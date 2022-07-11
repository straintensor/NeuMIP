//
// Created by alex on 2/23/21.
//

#ifndef CPPACC_TORCHACTIVEMODEL_H
#define CPPACC_TORCHACTIVEMODEL_H


#include <boost/python.hpp>
#include <torch/extension.h>
#include "gpu_torch_utils.h"
#include "gpuhelper/GPUhelper.h"
#include "storage_model.h"
#include "active_model.h"
#include <vector>

class TorchActiveModel {
public:

    std::vector<gpuObjectArr<float>> buffers_data;
    gpuObjectArr<ActiveNeuralTexture> texture_array;
    gpuObjectArr<GPUActiveModel> gam_gpu;

    TorchActiveModel(const char *path);
    TorchActiveModel(const TorchActiveModel&) = delete;
    torch::Tensor evaluate(torch::Tensor deferred_buffer, float zoom_in_multiplier);


};

#endif //CPPACC_TORCHACTIVEMODEL_H
