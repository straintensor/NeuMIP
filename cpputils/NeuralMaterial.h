//
// Created by alex on 3/23/21.
//

#ifndef CPPACC_NEURALMATERIAL_H
#define CPPACC_NEURALMATERIAL_H


#include "storage_model.h"
#define NO_PYTHON
#include "storage_model_impl.h"
#include "active_model.h"
//#include "active_model.cpp"

class NeuralMaterial {
public:
    std::vector<gpuObjectArr<float>> buffers_data;
    gpuObjectArr<ActiveNeuralTexture> texture_array;
    gpuObjectArr<GPUActiveModel> gam_gpu;

    NeuralMaterial() {};

    void init(const char *path, GPUActiveModel &cpu_gam) {


        StorageModel storage_model;
        storage_model.open(path);
        cpu_gam.init_networks(storage_model);


        cpu_gam.num_lod = storage_model.neural_mipmap_texture.size();

        cpu_gam.neural_offset_texture = ActiveNeuralTexture(storage_model.neural_offset_texture);
        std::cout << "storage_model.neural_offset_texture.data.size " << storage_model.neural_offset_texture.data.size() << std::endl;
        std::cout << "last" << storage_model.neural_offset_texture.data[storage_model.neural_offset_texture.data.size() - 1] << std::endl;

        buffers_data.emplace_back(storage_model.neural_offset_texture.data);
        cpu_gam.neural_offset_texture.data = buffers_data.back().get_cuda_ptr();


        std::vector<ActiveNeuralTexture> vacm(cpu_gam.num_lod);
        for (int i = 0; i < cpu_gam.num_lod; i++) {
            vacm[i] = ActiveNeuralTexture(storage_model.neural_mipmap_texture[i]);
            buffers_data.emplace_back(storage_model.neural_mipmap_texture[i].data);
            vacm[i].data = buffers_data.back().get_cuda_ptr();
        }

    //    printf("pp %p\n", vacm[0].data);

        texture_array = gpuObjectArr<ActiveNeuralTexture>(vacm);

        cpu_gam.neural_mipmap_texture = texture_array.get_cuda_ptr();



    }
    NeuralMaterial(const NeuralMaterial&) = delete;
};





#endif //CPPACC_NEURALMATERIAL_H
