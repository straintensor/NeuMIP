import torch
import utils
import numpy as np

import neurtext

def get_padding_name(tiled):
    if tiled:
        return "repeat"
    else:
        return "border"

class MipmapTexture:
    def __init__(self, nms=None, number_mip_maps_levels=None, num_ch=None, resolution=None, base_radius=None, tiled=None,
                 text=None
                 ):

        self.base_radius = base_radius
        self.number_mip_maps_levels = number_mip_maps_levels
        self.tiled = tiled
        self.mipmap_textures = []

        if text is not None:
            self.init_from_existing(text)
            return

        self.num_ch = num_ch

        base_resolution = resolution
        if True:
            for level in range(self.number_mip_maps_levels):
                resolution = np.array(list(base_resolution))
                resolution = resolution / 2 ** level
                resolution = [self.num_ch] + list(resolution.astype(int))

                text = torch.randn(resolution, **nms.live.get_type_device(), requires_grad=True)
                text = text*.1
                text = text.detach().requires_grad_()

                self.mipmap_textures.append(text)
        else:
            resolution = np.array(list(base_resolution))
            resolution = [self.num_ch] + list(resolution.astype(int))

            text = torch.randn(resolution, **nms.live.get_type_device(), requires_grad=True)
            blur_size = int(32)
            text.data[...] = utils.tensor.blur(blur_size, text.unsqueeze(0), True, sm=3)[0, :, :, :]  # * blur_size

            self.mipmap_textures.append(text)
            self.build_pyramid()




    def init_from_existing(self, text):
        text = text.detach().requires_grad_()

        self.num_ch = text.shape[0]

        self.mipmap_textures.append(text)
        self.build_pyramid()

    def build_pyramid(self):

        for level in range(self.number_mip_maps_levels - 1):
            with torch.no_grad():
                text = torch.nn.functional.interpolate(self.mipmap_textures[level-1].unsqueeze(0),
                                                       scale_factor=(.5, .5), mode='area')[0, ...]
            text = text.detach().requires_grad_()

            self.mipmap_textures.append(text)

    def recreate_pyramid(self):
        for level in range(self.number_mip_maps_levels - 1):
            with torch.no_grad():
                text = torch.nn.functional.interpolate(self.mipmap_textures[level-1].unsqueeze(0),
                                                       scale_factor=(.5, .5), mode='area')[0, ...]

                text = text.detach().requires_grad_()

                self.mipmap_textures[level][...] = text[...]

    def get_num_mipmap(self):
        return len(self.mipmap_textures)

    def get_params(self):
        return [x for x in self.mipmap_textures]

    def get_neural_texture_vis(self, index, mipmap_level_id):
        mipmap_level_id = int(np.round(mipmap_level_id))

        mipmap_texture = self.mipmap_textures[mipmap_level_id]

        scale_factor = 2 ** mipmap_level_id
        mipmap_texture = torch.nn.functional.interpolate(mipmap_texture.unsqueeze(0),
                                                         scale_factor=(scale_factor, scale_factor), mode='area')[0, ...]

        return neurtext.get_neural_texture_vis(mipmap_texture, index)

    def fuse_blur(self):
        # return False
        weight = .1
        with torch.no_grad():
            for idx, mipmap_texture in enumerate(self.mipmap_textures):
                mipmap_texture_new = utils.tensor.blur(1.03, mipmap_texture.unsqueeze(0), True)[0, :, :, :]

                mipmap_texture = mipmap_texture * (1 - weight) + mipmap_texture_new * weight

                self.mipmap_textures[idx][...] = mipmap_texture[...]


    def get_numpy_array(self, sigma):
        out_text = []

        for idx, text in enumerate(self.mipmap_textures):
            text = text.unsqueeze(0)
            text = utils.tensor.blur(sigma / 2 ** idx, text, True)
            text = text[0,:,:,:]
            text = text.permute(1,2,0)
            text = text.data.cpu().numpy()
            out_text.append(text)

        return out_text



    def evaluate_at_points(self, mimpap_type, cur_location, query_radius, sigma, level_id, eval_output=None,
                           max_level=-1):

        if max_level == -1:
            max_level = self.get_num_mipmap()


        mipmap_textures = self.mipmap_textures


        lookup_textures = []

        for idx in range(max_level):

            neural_texture = mipmap_textures[idx].unsqueeze(0)
            neural_texture = utils.tensor.blur(sigma / 2 ** idx, neural_texture, True)

            neural_texture = neural_texture.repeat([cur_location.shape[0], 1, 1, 1])

            min_size = 64
            while neural_texture.shape[-1] < min_size  and False:
                radius = 1
                scale_factor = radius * 4

                def pad_left(x):
                    # x = torch.cat([x, x[:, :, :, :1]], -1)
                    # x = torch.cat([x, x[:, :, :1, :]], -2)
                    return x

                # orig_text = neural_texture
                neural_texture = torch.nn.functional.interpolate(pad_left(neural_texture),
                                                                 scale_factor=(scale_factor, scale_factor),
                                                                 mode='bilinear', align_corners=False)

                #neural_texture = neural_texture[:, :, radius:-(radius + 1), radius:-(radius + 1)]

                # print(neural_texture.shape)

            lookup_texture = utils.tensor.grid_sample(
                neural_texture,
                cur_location,
                'bilinear',
                padding_mode=get_padding_name(self.tiled),
                compensate=mimpap_type == 0
            )
            #return lookup_texture
            lookup_textures.append(lookup_texture)


        with torch.no_grad():
            if query_radius is not None:
                if self.base_radius == 0:
                    level_id = torch.zeros_like(query_radius)
                else:
                    level_id = torch.log2(query_radius / self.base_radius)  # FIXME
                    level_id = level_id.clamp(min=0, max=max_level - 1)

                eval_output.level_id = level_id
            else:
                level_id = level_id

            level_id_base = level_id.floor()

            level_value = 1. - (level_id - level_id_base)

            level_id_next = (level_id_base + 1).clamp(min=0, max=max_level - 1).long()
            level_id_base = level_id_base.clamp(min=0, max=max_level - 1).long()

        total_lookup_texture = 0

        for idx, texture in enumerate(lookup_textures):
            total_lookup_texture = total_lookup_texture + (level_id_base == idx) * texture * level_value
            total_lookup_texture = total_lookup_texture + (level_id_next == idx) * texture * (1. - level_value)


        del idx, texture

        return total_lookup_texture