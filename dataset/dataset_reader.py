import torch

import h5py
import numpy as np
import os
import path_config
import dataset.rays as dataset_rays

class DatasetMulti(torch.utils.data.Dataset):
    def __init__(self, paths, *args, **kwargs):


        self.base_path = None

        self.datasets = []


        for cur_base_path in paths:
            if os.path.isdir(cur_base_path):
                for root, dirs, files in os.walk(cur_base_path):
                    for file in files:
                        if file.endswith(".pth"):
                            cur_path = os.path.join(cur_base_path, file)
                            dataset = Dataset(cur_path,*args, **kwargs)
                            if dataset.is_valid():
                                self.datasets.append(dataset)

            else:

                self.datasets.append(Dataset(cur_base_path,*args, **kwargs))


        dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.num_elements = sum(dataset_sizes)
        self.cum_size = np.cumsum(np.array(dataset_sizes))

    def __getitem__(self, item_id):
        dataset_id = np.searchsorted(self.cum_size, item_id, side='right')

        if dataset_id > 0:
            item_id = item_id - self.cum_size[dataset_id-1]

        return self.datasets[dataset_id][item_id]

    def __len__(self):
        return self.num_elements

    def resolution(self):
        return self.datasets[0].resolution()



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, get_rays_buffer=False, cosine_mult=False, boost=1.0):
        super(Dataset, self).__init__()

        if path_config.copy_dataset:
            import tempfile
            import shutil

            tfile_name =os.path.join(tempfile.gettempdir(), os.path.basename(path))


            shutil.copy2(path, tfile_name)
            print("Copying to ", tfile_name)

            path = tfile_name



        self.path = path
        self.get_rays_buffer = get_rays_buffer

        self.cosine_mult =cosine_mult
        self.boost = boost

        dataset = self.open_dataset()

        self.num_elements = dataset["ground_color"].shape[0]


    def open_dataset(self):
        return h5py.File(self.path, 'r')

    def __len__(self):
        return self.num_elements

    def resolution(self):
        dataset = self.open_dataset()
        return tuple(dataset["ground_color"].shape[1:3])

    def is_valid(self):
        valid = True

        dataset = self.open_dataset()

        if "base_radius" not in dataset.attrs:
            valid = False

        if not valid:
            print("Error with " + self.path)


        return valid

    def calc_average(self):
        valid_pixels = 0
        picture_sum = 0
        print(len(self))

        max_img = None

        for idx in range(len(self)): #len(self)
            buffer = self[idx]

            mask = 1

            buffer.light_dir = buffer.light_dir# -.8
            radius2 = buffer.light_dir*buffer.light_dir
            radius2 = radius2[:,:,0:1] + radius2[:,:,1:2]

            mask = mask*(radius2 < .2)

            buffer.camera_dir = buffer.camera_dir #+ .4
            radius22 = buffer.camera_dir*buffer.camera_dir
            radius22 = radius22[:,:,0:1] + radius22[:,:,1:2]

            mask = mask * (radius22 < .2)
            mask = 1

            #valid_radius = buffer.camera_query_radius /buffer.base_radius <= 1.1
            valid_radius = 1
            mask = mask * valid_radius

            valid_pixels = valid_pixels + mask
            picture_sum = picture_sum + buffer.colors*mask

            if max_img is None:
                max_img = buffer.colors
            else:
                max_img = np.minimum(buffer.colors, max_img)

        valid_pixels = valid_pixels + (valid_pixels==0)
        picture = picture_sum/valid_pixels
        #picture[valid_pixels == 0] = 0
        import utils
        utils.tensor.displays(picture)
        utils.tensor.displays(max_img/10)

        print("Done")


    def get_height_width(self):

        dataset = self.open_dataset()

        height, width = dataset["ground_camera_dir"].shape[1:3]

        return height, width


    def get_location(self, loc_j, loc_i):
        dataset = self.open_dataset()

       # num = 2

        ground_color = dataset["ground_color"][:, loc_j, loc_i, :]
        ground_light = dataset["ground_light"][:, loc_j, loc_i, :]
        ground_camera_dir = dataset["ground_camera_dir"][:, loc_j, loc_i, :]

        def convert_buffer(x):
            return torch.Tensor(x).float()#.permute([2,0,1])

        ground_color = convert_buffer(ground_color)
        ground_camera_dir = convert_buffer(ground_camera_dir[:,  :2])
        ground_light = convert_buffer(ground_light[:,  :2])


        return ground_color, ground_camera_dir, ground_light

    def __getitem__(self, item_id):
        dataset = self.open_dataset()

        ground_color = dataset["ground_color"][item_id, ...]
        ground_light = dataset["ground_light"][item_id, ...]
        ground_camera_dir = dataset["ground_camera_dir"][item_id, ...]

        def convert_buffer(x):
            return torch.Tensor(x).float().permute([2,0,1])

        ground_color = convert_buffer(ground_color)
        ground_camera_dir = convert_buffer(ground_camera_dir[:, :, :2])
        ground_light = convert_buffer(ground_light[:, :, :2])

        location = convert_buffer(dataset["ground_camera_target_loc"][item_id, ...])

        #import utils
        #location = utils.tensor.fmod1_1(location - .5)

        camera_query_radius = convert_buffer(dataset["ground_camera_query_radius"][item_id, ...])

        if "base_radius" in dataset.attrs:
            base_radius = dataset.attrs["base_radius"]
        else:
            base_radius = None


        input = torch.cat([ground_camera_dir, ground_light], dim=-3)



        if self.cosine_mult:
            radius2 = ground_light[0:1, :, :] * ground_light[0:1, :, :] + ground_light[1:2, :, :] * ground_light[1:2, :,:]

            ground_color = ground_color*torch.sqrt(1-radius2)

        if self.boost != 1:
            ground_color = ground_color*self.boost

        ground_color = torch.clamp(ground_color, 0, 1e3)



        if self.get_rays_buffer:
            buffer = dataset_rays.Buffers()

            def get_numpy(x: torch.Tensor):
                x = x.data.cpu().numpy()
                x = x.transpose(1,2,0)
                return x

            buffer.base_radius = base_radius
            buffer.camera_query_radius = get_numpy(camera_query_radius)
            buffer.camera_target_loc = get_numpy(location)

            buffer.colors = get_numpy(ground_color)

            buffer.camera_dir = get_numpy(ground_camera_dir)
            buffer.light_dir = get_numpy(ground_light)

            if "ground_valid" in dataset:
                buffer.valid_pixels = np.array(dataset["ground_valid"][item_id, ...], dtype=np.float32)

            if "ground_light_multiplier" in dataset:
                buffer.light_multiplier = np.array(dataset["ground_light_multiplier"][item_id, ...], dtype=np.float32)

            if True:
                pass
                # buffer.light_multiplier[...] = 2
                # buffer.valid_pixels[...] = 1
                # buffer.camera_query_radius[...] = 0
                #
                # import utils
                # utils.tensor.displays(buffer.light_dir*.5+.5)
                #
                # #buffer.light_dir[...] = 0
                # buffer.camera_dir[...] = 0
                # buffer.colors[...] = 0
                # a = buffer.camera_target_loc[:,:,0]+0
                # buffer.camera_target_loc[:,:,0] = buffer.camera_target_loc[:,:,1]
                # buffer.camera_target_loc[:, :, 1] = a

                # buffer.camera_target_loc[:, :, 0] =buffer.camera_target_loc[:, :, 0]#*.75
                # buffer.camera_target_loc[:, :, 1] =buffer.camera_target_loc[:, :, 1]#*.75
                #


                # import utils
                #
                # loc =  utils.tensor.generate_grid(512,512).data.cpu().numpy()[0,...].transpose(1,2,0)
                # buffer.camera_target_loc = loc

                #buffer.camera_target_loc =*.1 + buffer.camera_target_loc
                #
                # utils.tensor.displays(buffer.camera_target_loc)
                # utils.tensor.displays(buffer.colors == 0)

            buffer.finilize()

            return buffer


        if base_radius is None:
            base_radius = 1 / ground_color.shape[-1] / 2

        return input, ground_color, location, camera_query_radius, base_radius


def main():
    path = os.path.join(path_config.path_dataset, "stylized_tortoise_shell_1601_ss.hdf5")#stylized_tortoise_shell_1501_ss_on.hdf5")
    dataset = Dataset(path, True)
    #dataset.calc_average()

    import utils
    for i in range(15):
        utils.tensor.displays(dataset[i].colors)

if __name__ == "__main__":
    main()