#ifndef CPPACC_ACTIVE_MODULE_H
#define CPPACC_ACTIVE_MODULE_H

#include "active_modules.h"
#include "storage_model.h"
#include "neural_tex.h"
#include <vector>
#include <string>

#define True true
#define False false

//NeuralTexture



template <class GTexture, class VGTexture>
class ActiveModel{
public:

    ActiveLinearModule<9,25,True,True> offset_network_mpl0;
    ActiveLinearModule<25,25,True,True> offset_network_mpl1;
    ActiveLinearModule<25,25,True,True> offset_network_mpl2;
    ActiveLinearModule<25,1,True,False> offset_network_mpl3;
    ActiveLinearModule<11,25,True,True> main_mpl0;
    ActiveLinearModule<25,25,True,True> main_mpl1;
    ActiveLinearModule<25,25,True,True> main_mpl2;
    ActiveLinearModule<25,6,True,False> main_mpl3;

//    ActiveLinearModule<9,16,True,True> offset_network_mpl0;
//    ActiveLinearModule<16,16,True,True> offset_network_mpl1;
//    ActiveLinearModule<16,16,True,True> offset_network_mpl2;
//    ActiveLinearModule<16,1,True,False> offset_network_mpl3;
//    ActiveLinearModule<11,16,True,True> main_mpl0;
//    ActiveLinearModule<16,16,True,True> main_mpl1;
//    ActiveLinearModule<16,16,True,True> main_mpl2;
//    ActiveLinearModule<16,6,True,False> main_mpl3;


    GTexture neural_offset_texture;
    VGTexture neural_mipmap_texture;

    int num_lod = 0;


    void init_networks(StorageModel &storage_model) {
        main_mpl0.load_from_storage_model(storage_model.linear_modules["main.mpl0"]);
        main_mpl1.load_from_storage_model(storage_model.linear_modules["main.mpl1"]);
        main_mpl2.load_from_storage_model(storage_model.linear_modules["main.mpl2"]);
        main_mpl3.load_from_storage_model(storage_model.linear_modules["main.mpl3"]);
        offset_network_mpl0.load_from_storage_model(storage_model.linear_modules["offset_network.mpl0"]);
        offset_network_mpl1.load_from_storage_model(storage_model.linear_modules["offset_network.mpl1"]);
        offset_network_mpl2.load_from_storage_model(storage_model.linear_modules["offset_network.mpl2"]);
        offset_network_mpl3.load_from_storage_model(storage_model.linear_modules["offset_network.mpl3"]);
    }

    void init(const std::string &path) {};


    __host__ __device__
    float get_lod(float patch_size) const {

        float pixel_size = 1.f/neural_mipmap_texture[0].width;
        float ratio = patch_size/pixel_size;
        if (ratio <= 1) {
            ratio = 1;
        }
        float lod = std::log2(ratio);
        lod = min(lod, (float)(num_lod - 1));
        return lod;
    }



    __host__ __device__ void forward_stage_1(float uv_x, float uv_y, float lod,
                                         float camera_dir_x, float camera_dir_y,
                                         float *__restrict__ out_vec) const {
        const int max_vector_size = 25;


        float vector1[max_vector_size];
        float vector2[max_vector_size];


//        out[0] = uv_x*.5f + .5f;
//        out[1] = uv_y*.5f + .5f;
//        return;

        get_texture_values<7>(neural_offset_texture, uv_x, uv_y, vector1);

        vector1[7] = camera_dir_x;
        vector1[8] = camera_dir_y;

//        out[0] =  vector1[0]+.5f;
//        out[1] =  vector1[1]+.5f;
//        out[2] =  vector1[2]+.5f;
//        return;

        offset_network_mpl0.forward(vector1, vector2);
//        out_vec[0] = vector2[0+0]+.5f;
//        out_vec[1] = vector2[1+0]+.5f;
//        out_vec[2] = vector2[2+0]+.5f;
//        return;

        offset_network_mpl1.forward(vector2, vector1);
        offset_network_mpl2.forward(vector1, vector2);
        offset_network_mpl3.forward(vector2, vector1);

//        out[0] = vector1[0]+.5f;
//        out[1] = 0.f;
//        out[2] = 0.f;
//        return;

        float depth = vector1[0];

        float ux_x_old = uv_x;
        float ux_y_old = uv_y;

        am_utils::shift_function(uv_x, uv_y, depth, camera_dir_x, camera_dir_y);
        // std::cout << "Size: " << neural_mipmap_texture.size() << std::endl;
        // std::cout << "Width: " << neural_mipmap_texture[0].width << std::endl;


//            out[0] = camera_dir_x+.5f;
//            out[1] = camera_dir_y+.5f;
//                out[2] = 0.f;
//        return;


//        out[0] = uv_x-ux_x_old+.5f;
//        out[1] = uv_y-ux_y_old+.5;
//        out[2] = 0.f;
//        return;


//        out[0] = depth+.5f;
//        return;



        //printf("%i\n", num_lod);
        get_texture_values_lod<7,VGTexture>(neural_mipmap_texture, num_lod, uv_x, uv_y, lod, &out_vec[4]);
    }

    __host__ __device__ void forward_stage_2(float camera_dir_x, float camera_dir_y,
                                             float light_dir_x, float light_dir_y,
                                             float *__restrict__ in_vec, float *__restrict__ out_vec) const {
        const int max_vector_size = 25;

        in_vec[0] =light_dir_x;
        in_vec[1] = light_dir_y;
        in_vec[2] = camera_dir_x;
        in_vec[3] = camera_dir_y;

        float vector1[max_vector_size];
        float vector2[max_vector_size];


        main_mpl0.forward(in_vec, vector2);
        main_mpl1.forward(vector2, vector1);
        main_mpl2.forward(vector1, vector2);
        main_mpl3.forward(vector2, vector1);


        // get_texture_values(neural_mipmap_texture[0], uv_x, uv_y, vector1);
        am_utils::yuv_to_rgb_linear(vector1, vector2);
        for (int i = 0; i < 3; i++) {
            out_vec[i] = vector2[i];//*.5+.5;
        }

    }


    __host__ __device__
    void forward(float uv_x, float uv_y, float lod,
                 float camera_dir_x, float camera_dir_y,
                 float light_dir_x, float light_dir_y, float *out) const {

        float temp_vec[7+4];
        forward_stage_1(uv_x, uv_y, lod,
                camera_dir_x, camera_dir_y,
                temp_vec);

        forward_stage_2(camera_dir_x, camera_dir_y,
                light_dir_x, light_dir_y, temp_vec, out);

    }


};
#undef True
#undef False

typedef ActiveModel<NeuralTexture, std::vector<NeuralTexture>> CPUActiveModel;
typedef ActiveModel<ActiveNeuralTexture, ActiveNeuralTexture*> GPUActiveModel;





#endif //CPPACC_ACTIVE_MODULE_H
