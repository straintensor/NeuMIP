
import numpy as np
import h5py
import path_config
import os
import dataset.sceneinfo
import dataset.rays
import utils
import argparse

import random
import string
import dataset.dataset_reader
import renderer
import h5py
import struct
import random
from tqdm import tqdm


class DirectTensor:
    def __init__(self, h5obj, name, shape_base, dtype, chunks):
        self.ds = h5obj.create_dataset(name, shape_base, dtype=dtype, chunks=chunks)

        self.dtype = dtype

    def __setitem__(self, args, val):
        self.ds[args] = val.astype(dtype=self.dtype)
        #self.ds[args] = val

    def __del__(self):
        pass


class DeferredTensor:
    def __init__(self, h5obj, name, shape_base, dtype, chunks):
        self.h5obj = h5obj
        self.name, self.shape_base, self.dtype, self.chunks = name, shape_base, dtype, chunks

        self.dtype = dtype

        self.narray = np.zeros(self.shape_base, dtype=dtype)

    def __setitem__(self, args, val):
        self.narray[args] = val.astype(dtype=self.dtype)


    def reshuffle(self, order):
        shape = self.narray.shape
        num_chan = shape[-1]

        #narray = self.narray.reshape(-1, num_chan)

        new_narray = []
        for idx in range(num_chan):
            cur_array =  self.narray[:,:,:,idx]

            cur_shape = cur_array.shape
            cur_array =cur_array.reshape(-1)
            cur_array = cur_array[order]
            cur_array = cur_array.reshape(*cur_shape)

            new_narray.append(cur_array)

        self.narray = np.ascontiguousarray(np.stack(new_narray, axis=-1))






    def __del__(self):
        self.ds = self.h5obj.create_dataset(self.name, self.shape_base, dtype=self.dtype, chunks=self.chunks, data=self.narray)


class Dataset_Generator:
    def __init__(self, path, _, args):
        del _

        self.args = args
        self.btf = args.btf
        self.pbrt_in = args.pbrt_in

        self.res = args.res

        if args.cres is not None:
            self.dataset_res  = args.cres
        else:
            self.dataset_res = self.res

        self.angular = args.angular

        self.dataset = h5py.File(path, 'w')

        if args.dataset_in:
            dataset_paths_in = os.path.join(path_config.path_dataset,  args.dataset_in)
            self.dataset_in = dataset.dataset_reader.DatasetMulti([dataset_paths_in], get_rays_buffer=True)
        else:
            self.dataset_in = None


        self.scene_info = dataset.sceneinfo.SceneInfo()

        if self.dataset_in is None:
            self.scene_info.resolution = list(args.res)
        else:
            self.scene_info.resolution = list(self.dataset_in.resolution())

        self.scene_info.num_level = args.num_level

        if self.scene_info.num_level == -1:
            self.scene_info.num_level = int(np.ceil(np.log2(np.max(self.scene_info.resolution))))

        self.scene_info.fixed_camera = args.fixed_camera
        self.scene_info.fixed_light = args.fixed_light
        self.scene_info.fixed_patch = args.fixed_patch
        self.scene_info.ubo2014 = args.ubo2014

        self.super_smooth = args.super_smooth
        if self.super_smooth :
            args.smooth = self.super_smooth

        self.scene_info.smooth = args.smooth


        self.scene_info.object_name = args.object
        self.scene_info.material_name = args.material


        self.scene_info.tilesize = args.tilesize

        self.deferred = args.deferred

        if self.dataset_in is None:
            self.num_images = args.num_images

            if self.deferred:
                self.num_images = 1
            if  self.angular:
                self.num_images = 1
        else:
            self.num_images = len(self.dataset_in)

        self.renderer = args.renderer

        self.shuffled_ids = list(range(int(np.ceil(np.power(self.num_images, 1/4))**4)))
        random.shuffle(self.shuffled_ids)


        self.displacement_factor = args.displacement_factor

        if self.btf:
            self.process_btf()

        if self.pbrt_in:
            self.process_pbrt_in(self.pbrt_in[0], self.pbrt_in[1])


    def process_btf(self):
        path = os.path.join(path_config.path_dataset,  self.btf)

        self.btf_h5 = h5py.File(path, 'r')

        print(list(self.btf_h5))


        self.num_images, height, width, _ = self.btf_h5["img"].shape

        #self.num_images = 7

        self.res = [height, width]
        self.dataset_res = self.res
        self.scene_info.resolution = self.res

        print("Done")


    def process_pbrt_in(self, path, path2):
        path_query = os.path.join(path_config.path_dataset_pbrt, path)
        path_query_rgb = os.path.join(path_config.path_dataset_pbrt, path2)

        def open_floats(path):
            return np.fromfile(path, np.float32)


        nD_loc = 7
        nD_rgb = 3

        num_sample_per_image = self.dataset_res[0] * self.dataset_res[1]

        query_loc = open_floats(path_query)
        query_rgb = open_floats(path_query_rgb)

        total_num_samples = int(len(query_loc)/nD_loc)
        self.num_images = int(total_num_samples/num_sample_per_image)

        query_loc = query_loc[:self.num_images*num_sample_per_image*nD_loc]
        query_rgb = query_rgb[:self.num_images*num_sample_per_image*nD_rgb]

        query_loc = query_loc.reshape(self.num_images, self.dataset_res[0], self.dataset_res[1], nD_loc)
        query_rgb = query_rgb.reshape(self.num_images, self.dataset_res[0], self.dataset_res[1], nD_rgb)

        self.query_loc = query_loc
        self.query_rgb = query_rgb





    def generate_pbrt(self, path):
        path = os.path.join(path_config.path_dataset_pbrt, path)

        pbrt_path = open(path, "wb")

        for idx in range(self.num_images):
            aux_buffers = dataset.rays.Buffers(self.scene_info,
                                           None, None, self.num_images)

            data = aux_buffers.get_pbrt()


            bin = struct.pack(len(data)*"f", *data)
            pbrt_path.write(bin)

        print(pbrt_path)




    def generate(self):

        global dataset
        base_type = np.float16

        if self.renderer == "mitsuba" or self.deferred:
            import dataset.generator_mitsuba
            mr = dataset.generator_mitsuba.MitsubaRenderer()
        else:
            assert False

        shape_base = [self.num_images ] + self.dataset_res


        chunk_size = [1] + self.dataset_res

        if self.btf:
            Tensor = DeferredTensor
        else:
            Tensor = DirectTensor




        def create_dataset(name, num_ch):
            return Tensor(self.dataset, name, shape_base + [num_ch], dtype=base_type, chunks=tuple(chunk_size + [num_ch]))

        ground_color = create_dataset('ground_color', 3)
        ground_camera_dir = create_dataset('ground_camera_dir', 2)
        ground_camera_target_loc =create_dataset('ground_camera_target_loc', 2)
        ground_camera_query_radius =  create_dataset('ground_camera_query_radius', 1)

        if self.deferred or self.dataset_in or self.angular:
            ground_valid = create_dataset('ground_valid', 1)
            ground_light_multiplier = create_dataset('ground_light_multiplier', 3) #TODO check
        else:
            ground_valid = None
            ground_light_multiplier = None


        ground_light = create_dataset('ground_light',2)


        #for idx, light_setup in enumerate(light_setups):
        seeds = np.random.random_integers(0, 1024*1024, self.num_images)


        if self.super_smooth:
            image_samples = self.scene_info.resolution[0]*self.scene_info.resolution[0]

            total_samples = (self.num_images) * image_samples
            cur_samples = 0
            rays_buffers = []

            possible_levels = np.arange(0, self.scene_info.num_level)
            levels_prob = 1.7**(possible_levels)
            levels_prob = levels_prob/levels_prob.sum()

            levels_prob += .5
            levels_prob = levels_prob/levels_prob.sum()

            area_of_levels = levels_prob*0




            progress = tqdm(total=total_samples)
            while cur_samples < total_samples:
                lod = np.random.choice(possible_levels, p=levels_prob)
                #print(lod)


                current_resolution = (np.array(self.scene_info.resolution)/2**lod).astype(int)

                area_of_levels[lod] += current_resolution[0]*current_resolution[1]

                ray_buffer = dataset.rays.Buffers(self.scene_info,
                                                       None, 0, self.num_images, lod=lod,
                                                  current_resolution=current_resolution)


                #print(ray_buffer.camera_origin.shape)

                ray_buffer.convert_to_line()
                rays_buffers.append(ray_buffer)

                base_radius = ray_buffer.base_radius
                #print("base_radius", base_radius)

                progress.update(ray_buffer.get_length())

                cur_samples += ray_buffer.get_length()

            def cat(x):
                return np.concatenate(x, axis=0)

            area_of_levels = area_of_levels/area_of_levels.sum()
            print("Areas: ", area_of_levels)

            camera_origin = cat([rb.camera_origin for rb in rays_buffers])
            camera_target_loc =  cat([rb.camera_target_loc for rb in rays_buffers])
            camera_dir =  cat([rb.camera_dir for rb in rays_buffers])
            camera_query_radius = cat([rb.camera_query_radius for rb in rays_buffers])
            light_dir = cat([rb.light_dir for rb in rays_buffers])


        progress = tqdm(total=self.num_images)

        for idx in range(self.num_images):

            deferred = False
            from_deferred = False

            if self.dataset_in is None:
                if self.deferred:
                    deferred = True
                    scene_info = renderer.setup_test1(True, self.scene_info.resolution)
                    if self.args.zoom_level is not None:
                        scene_info.zoom_level = self.args.zoom_level



                    standard_rendering, output_tensor = renderer.render_mitsuba(scene_info, mr)
                    #print("standard_rendering mean: ", standard_rendering.mean() )
                    #utils.tensor.displays(standard_rendering)

                    aux_buffers = renderer.convert_to_rays_buffer(output_tensor, scene_info.zoom_in_multiplier)
                    if False:
                        aux_buffers.display()
                    buffer = standard_rendering
                elif self.angular:
                    deferred = True
                    angle_type = int(self.angular[0])
                    loc_x = float(self.angular[1])
                    loc_y = float(self.angular[2])
                    dir_x = float(self.angular[3])
                    dir_y = float(self.angular[4])
                    lod = float(self.angular[5])


                    aux_buffers = dataset.rays.Buffers()
                    aux_buffers.generate_angular(self.scene_info, self.dataset_res, angle_type,
                                                 [loc_x, loc_y], [dir_x, dir_y], lod)
                    buffer = np.zeros_like(aux_buffers.camera_target_loc)
                elif self.btf:
                    deferred = True

                    aux_buffers = dataset.rays.Buffers(self.scene_info)
                    aux_buffers.generate_from_btf(self.btf_h5['img'][idx, :,:,:],
                                                  self.btf_h5['phiVthetaVphiLthetaL'][idx, :, :, :]
                                                  )
                    buffer = aux_buffers.colors

                elif self.pbrt_in:
                    deferred = True

                    aux_buffers = dataset.rays.Buffers(self.scene_info)
                    aux_buffers.generate_from_pbrt(self.query_rgb[idx,:,:,:], self.query_loc[idx,:,:,:]
                                                  )
                    buffer = aux_buffers.colors

                else:
                    if self.super_smooth:
                        aux_buffers = dataset.rays.Buffers()

                        def convert_to_image(x):
                            x = x[idx*image_samples:(idx+1)*image_samples,:]
                            # x = x.reshape(-1)
                            # print(x.shape)
                            x = x.reshape(self.dataset_res[0], self.scene_info.resolution[0], -1)
                            return x

                        aux_buffers.camera_origin = convert_to_image(camera_origin)
                        aux_buffers.camera_target_loc = convert_to_image(camera_target_loc)
                        aux_buffers.camera_dir = convert_to_image(camera_dir)
                        aux_buffers.camera_query_radius = convert_to_image(camera_query_radius)
                        aux_buffers.light_dir = convert_to_image(light_dir)
                        aux_buffers.aux_buffers = base_radius

                        aux_buffers.resolution = self.scene_info.resolution


                    else:
                        aux_buffers = dataset.rays.Buffers(self.scene_info,
                                                           None, self.shuffled_ids[idx], self.num_images)
            else:
                aux_buffers = self.dataset_in[idx]
                #aux_buffers.display()
                from_deferred = True

            if not deferred:
                if self.renderer == "mitsuba":

                    #utils.tensor.displays(aux_buffers.camera_target_loc*.5+.5)
                    buffer = mr.render( mr.create_scene(self.scene_info, buffers=aux_buffers)[0])

                    buffer = buffer / (np.abs(aux_buffers.light_dir[:,:,2:3])+1e-6)




            assert len(buffer.shape) == 3
            assert buffer.shape[0] == aux_buffers.resolution[0]
            assert buffer.shape[1] == aux_buffers.resolution[1]
            assert buffer.shape[2] == 3


            if from_deferred:
                buffer = buffer*aux_buffers.valid_pixels*aux_buffers.light_multiplier

            def sanatize(x):
                return np.nan_to_num(x, nan=0, posinf=0, neginf=0)


            ground_light[idx, ...] = sanatize(aux_buffers.light_dir[:, :, :2])

            ground_color[idx, :] = buffer
            ground_camera_dir[idx, ...] = sanatize(aux_buffers.camera_dir[:, :, :2])
            ground_camera_target_loc[idx, ...] = sanatize(aux_buffers.camera_target_loc[:, :, :2])
            ground_camera_query_radius[idx, ...] = sanatize(aux_buffers.camera_query_radius[:, :, :1])

            if idx == 0 and aux_buffers.base_radius is not None:
                self.dataset.attrs["base_radius"] = float(aux_buffers.base_radius)

            if ground_valid is not None:
                if aux_buffers.valid_pixels is not None:
                    ground_valid[idx, ...] = sanatize(aux_buffers.valid_pixels[:, :, :1])
                if from_deferred:
                    ground_valid[idx, ...] = aux_buffers.valid_pixels



            if ground_light_multiplier is not None:
                if aux_buffers.light_multiplier is not None:
                    ground_light_multiplier[idx, ...] = sanatize(aux_buffers.light_multiplier[:, :, :3])
                if from_deferred:
                    ground_light_multiplier[idx, ...] = aux_buffers.light_multiplier


            #
            #utils.tensor.displays(buffer)
            # exit()
            progress.update()

        if self.btf:
            num_elements = self.num_images*self.scene_info.resolution[0]*self.scene_info.resolution[1]

            order = np.array(list(range(int(num_elements))))
            np.random.shuffle(order)

            ground_color.reshuffle(order)
            ground_camera_dir.reshuffle(order)
            ground_camera_target_loc.reshuffle(order)
            ground_camera_query_radius.reshuffle(order)
            ground_light.reshuffle(order)


def main(dataset_name, args):
    path = os.path.join(path_config.path_dataset, dataset_name)

    if args.jobs:
        os.makedirs(path, exist_ok=True)

        filename  = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + ".pth"
        path = os.path.join(path, filename)

    data_gen = Dataset_Generator(path, 16, args)

    if not args.pbrt:
        data_gen.generate()
    else:
        data_gen.generate_pbrt(args.pbrt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument("--fixed_camera", action='store_true')
    parser.add_argument("--fixed_light", action='store_true')
    parser.add_argument("--fixed_patch", action='store_true')

    parser.add_argument("--ubo2014", action='store_true')

    parser.add_argument("--smooth", action='store_true')

    parser.add_argument("--super_smooth", action='store_true')


    parser.add_argument("--dataset_in", type=str)

    parser.add_argument("--renderer", choices=["mitsuba"], default="mitsuba")

    parser.add_argument("--object", type=str)
    parser.add_argument("--material", type=str)
    parser.add_argument("--num_images", type=int, default=10)

    parser.add_argument("--tilesize", type=int, default=1)

    parser.add_argument("--jobs", action='store_true')

    parser.add_argument("--deferred", action='store_true')

    parser.add_argument("--num_level", type=int, default=-1)

    parser.add_argument("--angular", type=str, default=None, nargs=6)

    parser.add_argument("--displacement_factor", type=float, default=None)

    parser.add_argument("--res", type=int, default=[512, 512], nargs=2)
    parser.add_argument("--cres", type=int, default=None, nargs=2)

    parser.add_argument("--pbrt", type=str, default=None)

    parser.add_argument("--pbrt_in", type=str, default=None, nargs=2)

    parser.add_argument("--btf", type=str, default=None)


    parser.add_argument("--zoom_level", type=float, default=None)


    args = parser.parse_args()

    main(args.dataset_name, args)