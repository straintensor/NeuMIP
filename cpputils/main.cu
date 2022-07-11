
#include <boost/python.hpp>
#include <torch/extension.h>
#include <vector>
#include "gpuhelper/GPUhelper.h"
#include "gpu_torch_utils.h"
#include "TorchActiveModel.h"
namespace bpy=boost::python;




class NeuralMipmapTexture {
public:
    texture<float, 2, cudaReadModeElementType> tex;





//
//    __global__ static void kernel (texture<float, 2, cudaReadModeElementType> tex, int m, int n)
//    {
//        int val;
//        for (int row = 0; row < m; row++) {
//            for (int col = 0; col < n; col++) {
//                val = tex2D (tex, col+0.5f, row+0.5f);
//                printf ("%3d  ", val);
//            }
//            printf ("\n");
//        }
//    }

    NeuralMipmapTexture() {}

    NeuralMipmapTexture(torch::Tensor tensor_in) {
        int m = 4; // height = #rows
        int n = 3; // width  = #columns
        size_t pitch, tex_ofs;
        float arr[4][3]= {{10, 11, 12},
                          {20, 21, 22},
                          {30, 31, 32},
                          {40, 41, 42}};


        float *arr_d = 0;
        CUDA_SAFE_CALL(cudaMallocPitch((void**)&arr_d,&pitch,n*sizeof(*arr_d),m));
        CUDA_SAFE_CALL(cudaMemcpy2D(arr_d, pitch, arr, n*sizeof(arr[0][0]),
                                    n*sizeof(arr[0][0]),m,cudaMemcpyHostToDevice));
        tex.normalized = false;
        CUDA_SAFE_CALL (cudaBindTexture2D (&tex_ofs, &tex, arr_d, &tex.channelDesc,
                                           n, m, pitch));

      //  kernel<<<1,1>>>(tex, m, n);
    }


};

NeuralMipmapTexture make_NMT(torch::Tensor tensor_in) {



    return NeuralMipmapTexture(tensor_in);
}









__device__ void get_texture_values(AccFloat3 &texture_tensor, float uv_x, float uv_y, float *data) {
    int text_width =  texture_tensor.size(2);
    int text_height =  texture_tensor.size(1);
    int text_num_ch =  texture_tensor.size(0);

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

    for (int ch = 0; ch < text_num_ch; ch++) {
        float val = (texture_tensor[ch][loc_text_0_y_low][loc_text_0_x_low] *(1.f - area_x_high) +
                texture_tensor[ch][loc_text_0_y_low][loc_text_0_x_high] *area_x_high)*(1.f - area_y_high)
                        +
                (texture_tensor[ch][loc_text_0_y_high][loc_text_0_x_low] *(1.f - area_x_high) +
                texture_tensor[ch][loc_text_0_y_high][loc_text_0_x_high] *area_x_high)*area_y_high;

        data[ch] = val;
    }
}

__global__ void cuda_kernel_texture_interpolate(
        AccFloat4 uv_tensor,
        AccFloat3 texture_tensor_0,
        AccFloat4 result_tensor,
        JobSize tensor_size) {

    JobSize tensor_loc;
    size_t loc_linear = blockIdx.x * blockDim.x + threadIdx.x;

    if (tensor_size.calc_position(tensor_loc, loc_linear)) {


        int text_num_ch =  texture_tensor_0.size(0);

        float uv_x = uv_tensor[tensor_loc.batch][0][tensor_loc.height][tensor_loc.width];
        float uv_y = uv_tensor[tensor_loc.batch][1][tensor_loc.height][tensor_loc.width];

        uv_x = (uv_x + 1.f)*.5f; //put between 0 and 1
        uv_y = (uv_y + 1.f)*.5f; //put between 0 and 1

        float data[7];

        get_texture_values(texture_tensor_0, uv_x, uv_y, data);

        for (int ch = 0; ch < text_num_ch; ch++) {
            result_tensor[tensor_loc.batch][ch][tensor_loc.height][tensor_loc.width] = data[ch];
        }

    }

}




__global__ void cuda_kernel_texture_interpolate_lod(
        AccFloat4 uv_tensor,
        AccFloat4 lod_tensor,
        int num_lod,
        AccFloat3 *texture_tensor,
        AccFloat4 result_tensor,
        JobSize tensor_size) {

    JobSize tensor_loc;
    size_t loc_linear = blockIdx.x * blockDim.x + threadIdx.x;

    if (tensor_size.calc_position(tensor_loc, loc_linear)) {


        int text_num_ch =  texture_tensor[0].size(0);

        float uv_x = uv_tensor[tensor_loc.batch][0][tensor_loc.height][tensor_loc.width];
        float uv_y = uv_tensor[tensor_loc.batch][1][tensor_loc.height][tensor_loc.width];

        uv_x = (uv_x + 1.f)*.5f; //put between 0 and 1
        uv_y = (uv_y + 1.f)*.5f; //put between 0 and 1

        float data_low[7];
        float data_high[7];

        float lod =  uv_tensor[tensor_loc.batch][0][tensor_loc.height][tensor_loc.width];

        lod = utils::clamp(lod, 0, num_lod-1);
        int lod_low = floor(lod);

        float lod_high_weight = lod - lod_low;
        int lod_high = min(lod_low+1,  num_lod-1);

        get_texture_values(texture_tensor[lod_low], uv_x, uv_y, data_low);
        get_texture_values(texture_tensor[lod_high], uv_x, uv_y, data_high);

        for (int ch = 0; ch < text_num_ch; ch++) {
            result_tensor[tensor_loc.batch][ch][tensor_loc.height][tensor_loc.width] =
                    data_low[ch]*(1.f - lod_high_weight) + data_high[ch]*lod_high_weight;
        }

    }

}


torch::Tensor texture_interpolate(torch::Tensor uv_tensor,  torch::Tensor texture_tensor_0) {

    CHECK_TENSOR_INPUT(uv_tensor);;
    CHECK_TENSOR_INPUT(texture_tensor_0);

    JobSize js;
    js.batch = uv_tensor.size(0);
    js.height = uv_tensor.size(2);
    js.width = uv_tensor.size(3);

    int num_ch = texture_tensor_0.size(0);


    auto options =
            torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .layout(torch::kStrided)
                    .device(torch::kCUDA, 0)
                    .requires_grad(false);

    torch::Tensor result_tensor = torch::empty({js.batch, num_ch, js.height, js.width}, options);

    const int block_size = 256;


    cuda_kernel_texture_interpolate<<<js.get_num_blocks(block_size), block_size>>> (
            uv_tensor.packed_accessor64<TFloat,4,torch::RestrictPtrTraits>(),
            texture_tensor_0.packed_accessor64<TFloat,3,torch::RestrictPtrTraits>(),
            result_tensor.packed_accessor64<TFloat,4,torch::RestrictPtrTraits>(),
            js
            );



    return result_tensor;
}


torch::Tensor texture_lod_interpolate(torch::Tensor uv_tensor,torch::Tensor lod_tensor,  bpy::list texture_tensors) {

    CHECK_TENSOR_INPUT(uv_tensor);;


    std::vector<AccFloat3> text_accessors;
    int num_lod = int(bpy::len(texture_tensors));

    for (int i = 0; i < num_lod; i++) {
        torch::Tensor text = bpy::extract<torch::Tensor>(texture_tensors[i]);
        CHECK_TENSOR_INPUT(text);
        text_accessors.push_back(text.packed_accessor64<TFloat,3,torch::RestrictPtrTraits>());
    }



    gpuObjectArr<AccFloat3> text_accessors_gpu(text_accessors);


    JobSize js;
    js.batch = uv_tensor.size(0);
    js.height = uv_tensor.size(2);
    js.width = uv_tensor.size(3);

    int num_ch = text_accessors[0].size(0);


    auto options =
            torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .layout(torch::kStrided)
                    .device(torch::kCUDA, 0)
                    .requires_grad(false);

    torch::Tensor result_tensor = torch::empty({js.batch, num_ch, js.height, js.width}, options);

    const int block_size = 256;




    cuda_kernel_texture_interpolate_lod<<<js.get_num_blocks(block_size), block_size>>> (
            uv_tensor.packed_accessor64<TFloat,4,torch::RestrictPtrTraits>(),
            lod_tensor.packed_accessor64<TFloat,4,torch::RestrictPtrTraits>(),
            num_lod,
            text_accessors_gpu.get_cuda_ptr(),
            result_tensor.packed_accessor64<TFloat,4,torch::RestrictPtrTraits>(),
            js
    );



    return result_tensor;
}



PYBIND11_MODULE(cppacc, m) {

    //m.def("make_NMT", make_NMT);
    m.def("texture_interpolate", texture_interpolate);

    py::class_<TorchActiveModel>(m, "TorchActiveModel")
            .def(py::init<const char *>())
            .def("evaluate", &TorchActiveModel::evaluate)
            ;


}