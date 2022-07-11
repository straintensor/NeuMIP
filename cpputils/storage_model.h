
#ifndef CPPACC_STORAGE_MODEL_H
#define CPPACC_STORAGE_MODEL_H

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/archives/portable_binary.hpp>
//#include <cereal/archives/binary.hpp>
//#include <boost/python.hpp>

#include <fstream>

namespace boost {

    namespace python {
        class list;
        namespace api  {
          class object;
        }
        using api::object;
        namespace numpy {
            class ndarray;
        }

    }
}


namespace bpy=boost::python;
namespace bnp = boost::python::numpy;


class NeuralTexture{
public:
    int32_t height, width, channel;
    std::vector<float> data;


    void set_from_numpy(bnp::ndarray  tensor);



    inline float get(size_t y, size_t x, size_t ch) const {
        size_t pos = (y*width + x) * channel + ch;
        return data[pos];
    }


    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(CEREAL_NVP(height), CEREAL_NVP(width), CEREAL_NVP(channel), CEREAL_NVP(data));
    }

};

class LinearModule {
public:
    int32_t ch_in = 0, ch_out = 0;
    int32_t has_bias = false;
    int32_t has_leru = false;

    std::vector<float> matrix;
    std::vector<float> bias;


    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(CEREAL_NVP(ch_in), CEREAL_NVP(ch_out), CEREAL_NVP(has_bias), CEREAL_NVP(has_leru), CEREAL_NVP(matrix), CEREAL_NVP(bias));
    }

};


class StorageModel {
public:

    NeuralTexture neural_offset_texture;
    std::vector<NeuralTexture> neural_mipmap_texture;

    std::map<std::string, LinearModule> linear_modules;
    std::string model_arch_name;

    void add_linear_module(const char * name, bnp::ndarray matrix, boost::python::object bias, bool has_leru);
    void add_linear_module_simple(const char * name) {};

    void save(const char *filename) {
        std::ofstream os(std::string(filename), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive( os );
        //cereal::BinaryOutputArchive archive( os );

        archive( *this );


        std::cout << "neural_mipmap_texture size " << neural_mipmap_texture.size() << std::endl;
    }
    void open(const char *filename) {

        std::ifstream in_stream(filename,  std::ios::in | std::ios::binary);
        std::istream &iin_stream =  in_stream;
//std::stringstream ss;

        cereal::PortableBinaryInputArchive iarchive(iin_stream);
       // cereal::BinaryInputArchive iarchive(iin_stream);

        iarchive( *this );

        std::cout << "neural_mipmap_texture size " << neural_mipmap_texture.size() << std::endl;

    }


    void set_nerual_offset_texture(bnp::ndarray  tensor);
    void set_nerual_mipmap_texture(bpy::list tensors);

    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(CEREAL_NVP(neural_offset_texture),  CEREAL_NVP(model_arch_name), CEREAL_NVP(neural_mipmap_texture), CEREAL_NVP(linear_modules));

    }

};


#endif //CPPACC_STORAGE_MODEL_H
