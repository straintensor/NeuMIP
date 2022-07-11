//
// Created by alex on 2/22/21.
//

#ifndef CPPACC_STORAGE_MODEL_IMPL_H
#define CPPACC_STORAGE_MODEL_IMPL_H


#include <vector>
#include <string>

#include <map>


#include <fstream>

#include "storage_model.h"








#ifndef NO_PYTHON
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>


void StorageModel::add_linear_module(const char * c_name, bnp::ndarray matrix, bpy::object bias, bool has_leru) {
    LinearModule linear_module;

   // extract<char const*>(s);

    const std::string name(c_name);//= py_name

    matrix = matrix.astype(bnp::dtype::get_builtin<float>());
    linear_module.ch_in = matrix.shape(1);
    linear_module.ch_out = matrix.shape(0);
    linear_module.has_leru = has_leru;


    matrix = matrix.reshape(boost::python::make_tuple(-1));

    linear_module.matrix.resize(linear_module.ch_in*linear_module.ch_out);

    for (int i = 0; i < linear_module.matrix.size(); i++) {
        linear_module.matrix[i] = bpy::extract<float>(matrix[i]);
    }



    if (!bias.is_none()) {
        linear_module.has_bias = true;

        bnp::ndarray bias_array = bpy::extract<bnp::ndarray>(bias);


        bias_array = bias_array.astype(bnp::dtype::get_builtin<float>());
        bias_array = bias_array.reshape(boost::python::make_tuple(-1));


        linear_module.bias.resize(linear_module.ch_out);

        for (int i = 0; i < linear_module.ch_out; i++) {
            linear_module.bias[i] = bpy::extract<float>(bias_array[i]);
        }

    }

    linear_modules[name] = linear_module;
}



void StorageModel::set_nerual_offset_texture(bnp::ndarray  tensor) {
    neural_offset_texture.set_from_numpy(tensor);
}
void StorageModel::set_nerual_mipmap_texture(bpy::list tensors) {
    int num_lod = int(bpy::len(tensors));

    neural_mipmap_texture.resize(num_lod);

    for (int i = 0; i < num_lod;  i++) {
        bnp::ndarray text = bpy::extract<bnp::ndarray>(tensors[i]);
        neural_mipmap_texture[i].set_from_numpy(text);
    }
}

void NeuralTexture::set_from_numpy(bnp::ndarray  tensor) {
    height = tensor.shape(0);
    width = tensor.shape(1);
    channel = tensor.shape(2);


    std::cout << "shape: " << height << " " << width << " " << channel << std::endl;

    //tensor = tensor.transpose();

    tensor = tensor.astype(bnp::dtype::get_builtin<float>());
    tensor = tensor.reshape(boost::python::make_tuple(-1));

    data.resize(height*width*channel);

    for (int i = 0; i < height*width*channel; i++) {
        data[i] = bpy::extract<float>(tensor[i]);
    }
}


BOOST_PYTHON_MODULE(storage_model) {
    bpy::class_<StorageModel>("StorageModel").
            def("add_linear_module", &StorageModel::add_linear_module).

            def("add_linear_module_simple", &StorageModel::add_linear_module_simple).
            def("set_nerual_offset_texture", &StorageModel::set_nerual_offset_texture).
            def("set_nerual_mipmap_texture", &StorageModel::set_nerual_mipmap_texture).
            def("save", &StorageModel::save);
}




#endif

#endif //CPPACC_STORAGE_MODEL_IMPL_H
