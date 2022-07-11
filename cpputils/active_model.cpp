#include "active_model.h"

template<>
void CPUActiveModel::init(const std::string &path) {
    StorageModel storage_model;
    storage_model.open(path.c_str());
    init_networks(storage_model);


    neural_offset_texture = storage_model.neural_offset_texture;
    neural_mipmap_texture = storage_model.neural_mipmap_texture;
    num_lod = neural_mipmap_texture.size();
}
