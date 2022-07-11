//
// Created by alex on 2/12/21.
//

#ifndef CPPACC_ACTIVE_MODULES_H
#define CPPACC_ACTIVE_MODULES_H

#include "storage_model.h"
#include "active_module_utils.h"
#include<iostream>
#include <math.h>


class ActiveNeuralTexture {
public:
    int height = 0, width = 0, channel = 0;
    int padding;

    float *data = nullptr;

    ActiveNeuralTexture() {}
    ActiveNeuralTexture(NeuralTexture &nt) {
        height = nt.height;
        width = nt.width;
        channel = nt.channel;
    }


    __host__ __device__
    inline float get(size_t y, size_t x, size_t ch) const {
        //printf("shape %i %i %i", height, width, channel);
        //printf("coord %i  %i %i\n", (int)y, (int)x, (int)ch);
        size_t pos = (y*width + x) * channel + ch;
        //size_t pos = (ch*width + x)*height + y;


        return data[pos];
    }


};





template <typename T>
void TAssert(T val, T ref, const char * error_message= nullptr) {
    if (val != ref) {
        if (error_message == nullptr) {
            error_message = "???";
        }
        std::cerr << "Error:\t" << val << " !=" << ref << " - " << error_message << std::endl;
        exit(-1);
    }

}

template <int CH_IN, int CH_OUT, bool HAS_BIAS, bool HAS_RELU>
class ActiveLinearModule {
public:
    float matrix[CH_OUT][CH_IN];

    float bias[CH_OUT];


    inline __host__ __device__ void forward(float * __restrict__ vector_in, float * __restrict__ vector_out) const {
        #pragma unroll
        for (int j = 0; j < CH_OUT; j++) {
            float val = 0.f;
            #pragma unroll
            for (int i = 0; i < CH_IN; i++) {
                val += matrix[j][i] * vector_in[i];
            }
            if (HAS_BIAS) {
                val += bias[j];
            }

            if (HAS_RELU) {
                if (val < 0) {
                    val = 0;
                }
            }
            vector_out[j] = val;
        }
    }

    __host__ void load_from_storage_model(LinearModule &lm){
        TAssert(lm.ch_in, CH_IN, "CH_IN");
        TAssert(lm.ch_out, CH_OUT, "CH_OUT");
        TAssert((bool)lm.has_bias, HAS_BIAS, "HAS_BIAS");
        TAssert((bool)lm.has_leru, HAS_RELU, "HAS_RELU");

        if (HAS_BIAS) {
            for (int i = 0; i < CH_OUT; i++) {
                bias[i] = lm.bias[i];
            }
        }

        int count = 0;


        for (int j = 0; j < CH_OUT; j++) {
        for (int i = 0; i < CH_IN; i++) {
                matrix[j][i] = lm.matrix[count];
                count++;
            }
        }


    }

};



class am_utils{
public:

static __host__ __device__
void shift_function(float &uv_x, float &uv_y, float depth, float dir_x, float dir_y) {
    float dir_z = sqrt(1.f - (dir_x*dir_x + dir_y*dir_y));
    dir_z = max(dir_z, .6f);

    float shift_x = depth * dir_x / dir_z *.1f;
    float shift_y = depth * dir_y / dir_z * .1f;

    uv_x += shift_x;
    uv_y += shift_y;
}



static __host__ __device__
inline float mask_sigmoid(float x) {
    return .1f + 1.f/(1.f + exp(-x));
}

static __host__ __device__
void yuv_to_rgb_linear(float * __restrict__ vec_in, float *__restrict__ vec_out) {

    vec_out[0] = vec_in[0] +                     1.13983f*vec_in[2];
    vec_out[1] =  vec_in[0] -0.39465f*vec_in[1]  -0.58060f*vec_in[2];
    vec_out[2] = vec_in[0] + 2.03211f*vec_in[1];

    for (int i = 0; i < 3; i++) {
        vec_out[i] = exp(vec_out[i]) - 1;
    }

    if (true) {
        for (int i = 0; i < 3; i++) {
            vec_out[i] *= am_utils::mask_sigmoid(vec_in[3+i]);
        }
    }

}


};


#endif //CPPACC_ACTIVE_MODULES_H
