import torch

def get_neural_texture_vis(mipmap_texture, index):


    mipmap_texture = mipmap_texture[:, :, :]

    neural_texture_slots = [None, None, None]
    for idx in range(3):
        cur_index = index + idx
        neural_texture_slots[cur_index % 3] = mipmap_texture[cur_index, :, :]

    mipmap_texture = torch.stack(neural_texture_slots, dim=0)

    mipmap_texture = mipmap_texture - mipmap_texture.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    # neural_texture = neural_texture / (neural_texture.std(dim=-1, keepdim=True).std(dim=-2, keepdim=True) + 1e-5)

    mipmap_texture = mipmap_texture * .5 + .5

    mipmap_texture = mipmap_texture.data.cpu().numpy()
    mipmap_texture = mipmap_texture.transpose([1, 2, 0])
    return mipmap_texture
