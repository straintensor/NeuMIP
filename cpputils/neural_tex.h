//
// Created by alex on 2/18/21.
//

#ifndef CPPACC_NEURAL_TEX_H
#define CPPACC_NEURAL_TEX_H

#include <vector>
#include <algorithm>

#include "active_module_utils.h"


template <int TEXT_NUM_CH, class T> inline
__host__ __device__
void get_texture_values(const T &texture_tensor, float uv_x, float uv_y, float * __restrict__ data) {

    uv_x = (uv_x + 1.f)*.5f; //put between 0 and 1
    uv_y = (uv_y + 1.f)*.5f; //put between 0 and 1

    int text_width =  texture_tensor.width;
    int text_height =  texture_tensor.height;
    int text_num_ch =  texture_tensor.channel;

   // printf("%p\n", texture_tensor.data);

    float uv_x_pixel = uv_x*text_width - .5;
    int loc_text_0_x_low = floor(uv_x_pixel);
    float area_x_high = uv_x_pixel- loc_text_0_x_low;
    loc_text_0_x_low = utils::pos_remainder(loc_text_0_x_low, text_width);
    int loc_text_0_x_high = (loc_text_0_x_low+1);
    loc_text_0_x_high = utils::pos_remainder(loc_text_0_x_high,text_width);


    float uv_y_pixel = uv_y*text_height - .5f;
    int loc_text_0_y_low = floor(uv_y_pixel);
    float area_y_high = uv_y_pixel  - loc_text_0_y_low;
    loc_text_0_y_low = utils::pos_remainder(loc_text_0_y_low, text_height);
    int loc_text_0_y_high = (loc_text_0_y_low+1);
    loc_text_0_y_high = utils::pos_remainder(loc_text_0_y_high,text_height);

    for (int ch = 0; ch < TEXT_NUM_CH; ch++) {
        float val = (texture_tensor.get(loc_text_0_y_low, loc_text_0_x_low, ch) *(1.f - area_x_high) +
                     texture_tensor.get(loc_text_0_y_low, loc_text_0_x_high, ch) *area_x_high)*(1.f - area_y_high)
                    +
                    (texture_tensor.get(loc_text_0_y_high, loc_text_0_x_low, ch) *(1.f - area_x_high) +
                     texture_tensor.get(loc_text_0_y_high, loc_text_0_x_high, ch) *area_x_high)*area_y_high;
       // float val = texture_tensor.get(loc_text_0_y_low, loc_text_0_x_low, ch);
        data[ch] = val;
    }
}






template <int TEXT_NUM_CH, class V>
__host__ __device__
void get_texture_values_lod(const V &texture_tensors, int num_lod, float uv_x, float uv_y, float lod, float *__restrict__ data) {
    float data_low[TEXT_NUM_CH];
    float data_high[TEXT_NUM_CH];



    lod = utils::clamp(lod, 0, num_lod-1.0001);
    int lod_low = floor(lod);

   // printf("lod_low %i\n", lod_low);


    float lod_high_weight = lod - lod_low;
    int lod_high = min(lod_low+1,  num_lod-1);

//printf("lod_low %i low %i high %i\n", lod_low, texture_tensors[lod_low].width, texture_tensors[lod_high].width);
//    printf("lod_low %i width %i hight %i channel %i data %p\n", lod_low, texture_tensors[0].width, texture_tensors[0].height,
//           texture_tensors[0].channel, texture_tensors[0].data);


    get_texture_values<TEXT_NUM_CH>(texture_tensors[lod_low], uv_x, uv_y, data_low);
    get_texture_values<TEXT_NUM_CH>(texture_tensors[lod_high], uv_x, uv_y, data_high);


    for (int i = 0; i < TEXT_NUM_CH; i++) {
        data[i] = data_low[i]* (1.f - lod_high_weight) + data_high[i] * lod_high_weight;
    }
}




#endif //CPPACC_NEURAL_TEX_H
