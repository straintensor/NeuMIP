//
// Created by alex on 2/23/21.
//

#include "TorchActiveModel.h"


__global__ void cuda_kernel_eval_model(AccFloat4 deferred_buffer,
    AccFloat4 result_tensor,
            GPUActiveModel *gam, int zoom_in_multiplier,
JobSize tensor_size) {
    JobSize tl;
    size_t loc_linear = blockIdx.x * blockDim.x + threadIdx.x;

    if (tensor_size.calc_position(tl, loc_linear)) {

        float valid_pixel = deferred_buffer[tl.batch][tl.height][tl.width][8];

        if (valid_pixel) {

            float camera_dir_x = deferred_buffer[tl.batch][tl.height][tl.width][0];
            float camera_dir_y = deferred_buffer[tl.batch][tl.height][tl.width][1];


            float light_dir_x = deferred_buffer[tl.batch][tl.height][tl.width][2];
            float light_dir_y = deferred_buffer[tl.batch][tl.height][tl.width][3];

            // between -1 and 1
            float location_x = deferred_buffer[tl.batch][tl.height][tl.width][4];
            float location_y = deferred_buffer[tl.batch][tl.height][tl.width][5];

            float radius_du = deferred_buffer[tl.batch][tl.height][tl.width][6];
            float radius_dv = deferred_buffer[tl.batch][tl.height][tl.width][7];

            float radius = (radius_du + radius_dv) * .5f;

            location_x *= zoom_in_multiplier;
            location_y *= zoom_in_multiplier;
            radius  *= zoom_in_multiplier;

//            location_x = location_x*2.f - 1.f;
//            location_y = location_y*2.f - 1.f;

           // location_x = -location_x;

            float lod = gam->get_lod(radius);

            float out[3];

            gam->forward(location_x, location_y, lod,
                         camera_dir_x, camera_dir_y,
                         light_dir_x, light_dir_y,
                         out);


            for (int i = 0; i < 3; i++) {
                result_tensor[tl.batch][tl.height][tl.width][i] = out[i];
            }

        }


    }


}


TorchActiveModel::TorchActiveModel(const char *path) {
        GPUActiveModel cpu_gam;

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

        printf("pp %p\n", vacm[0].data);

        texture_array = gpuObjectArr<ActiveNeuralTexture>(vacm);

        cpu_gam.neural_mipmap_texture = texture_array.get_cuda_ptr();


        gam_gpu = gpuObjectArr<GPUActiveModel>(1, &cpu_gam);
}




torch::Tensor TorchActiveModel::evaluate(torch::Tensor deferred_buffer, float zoom_in_multiplier) {

        CHECK_TENSOR_INPUT(deferred_buffer);



        JobSize js;
        js.batch = deferred_buffer.size(0);
        js.height = deferred_buffer.size(1);
        js.width = deferred_buffer.size(2);

        auto options =
                torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(torch::kStrided)
                        .device(torch::kCUDA, 0)
                        .requires_grad(false);


        torch::Tensor result_tensor = torch::empty({js.batch,  js.height, js.width, 3}, options);


        const int block_size = 256;

        cuda_kernel_eval_model<<<js.get_num_blocks(block_size), block_size>>>(
                deferred_buffer.packed_accessor64<TFloat, 4, torch::RestrictPtrTraits>(),
                result_tensor.packed_accessor64<TFloat, 4, torch::RestrictPtrTraits>(),
                gam_gpu.get_cuda_ptr(), zoom_in_multiplier,
                js);

        //cudaDeviceSynchronize();
        return result_tensor;
}
