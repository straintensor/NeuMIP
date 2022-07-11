import torch

import numpy as np
import os

from decorator import n_args

import path_config
import utils
import error_metric
import dataset.rays
import renderer
import aux_info
import cpputils

def add_param_CommonRenderer(parser):
    parser.add_argument('--ren_seq', type=str, nargs=5)
    parser.add_argument('--render', help='visualize', action='store_true')
    parser.add_argument('--def_dataset', help='dataset path for rendering', type=str, default=None)
    parser.add_argument('--out', help='out name', type=str, default="default")


    parser.add_argument('--zoom_level', type=float, default=None)
    parser.add_argument('--action_name', type=str, default=None)
    parser.add_argument('--spp', type=int, default=None)
    parser.add_argument('--timestamp', type=float, default=None)
    parser.add_argument('--light_pan', type=float, default=None)

    parser.add_argument('--png_only', help='png_only', action='store_true')

    parser.add_argument("--cpp_model", help='cpp_model', type=str, default=None)



class CommonRenderer:

    def render_sequence(self):
        export_img = True

        frame_id = int(self.args.ren_seq[0])
        num_frames = int(self.args.ren_seq[1])
        file_name = self.args.ren_seq[2]

        action_name = self.args.ren_seq[3]
        hero_object  = self.args.ren_seq[4]

        video_file_name = os.path.join(path_config.path_videos, file_name)
        if export_img:
            os.makedirs(video_file_name, exist_ok=True)


        #num_frames = 40

        timestamps = np.linspace(0, 1, num_frames)


        if not export_img:
            import imageio
            writer = imageio.get_writer(video_file_name, fps=20)

        buffers = []


        #for frame_id, timestamp in enumerate(timestamps):
        timestamp = timestamps[frame_id]
        print("\n", timestamp)
        buffer = self.render(timestamp, get_buffer=True, frame_id=frame_id, action_name=action_name, hero_object=hero_object)


        if export_img:
            image_path = "exr_{:04d}.exr".format(frame_id)
            image_path = os.path.join(video_file_name, image_path)
            utils.tensor.save_exr(buffer, image_path)


            image_path = "png_{:04d}.png".format(frame_id)
            image_path = os.path.join(video_file_name, image_path)


        else:
            buffer = utils.tensor.to_output_format(buffer)
            buffers.append(buffer)

        if not export_img:
            buffers = buffers + list(reversed(buffers))[1:-1]

            zoom_factor = 2

            for buffer in buffers:
                buffer = (buffer*255).astype('uint8')
                buffer = np.repeat(buffer, zoom_factor, 0)
                buffer = np.repeat(buffer, zoom_factor, 1)

                writer.append_data(buffer)

            writer.close()


    def cpp_evaluate2(self, datainput):
        #print("Evaluating")

        #print(datainput)

        datainput = torch.Tensor(datainput).float()
        datainput = datainput.unsqueeze(0)
        datainput = datainput.permute(0, 3, 1, 2)

        datainput = self.to_device(datainput)

        result, eval_output = self.brdf_eval(datainput)

        result = result[0, :, :, :]

        result = result.permute(1,2,0)
        result = result.data.cpu().numpy()
        result = np.ascontiguousarray(result)

        #print("Evaluating - done")

        return result

    def brdf_eval(self, output_tensor=None, rays_buffer: dataset.rays.Buffers=None):
        if output_tensor is not None:

            camera_dir = output_tensor[:, 0:2, :, :]
            light_dir = output_tensor[:, 2:4, :, :]
            location = output_tensor[:, 4:6, :, :]

            #print("Test")

            # camera_dir = utils.la4.vector_clamp(camera_dir, .85)
            # light_dir = utils.la4.vector_clamp(light_dir, .85)

            radius_du = output_tensor[:, 8:9, :, :]
            radius_dv = output_tensor[:, 9:10, :, :]

            valid_pixels =output_tensor[:, 7:8, :, :]

            radius = ((radius_du + radius_dv) / 2.0)

            #zoom_in_multiplier = 6*2
            zoom_in_multiplier = 12
            zoom_in_multiplier = 1

            zoom_in_multiplier = 6

            location = location*zoom_in_multiplier
            radius = radius*zoom_in_multiplier

            location = utils.tensor.fmod(location)# handle negative

            location = (location * 2 - 1)

            if False: # something weird
                location[:, 0:1, :, :] = -location[:, 0:1, :, :]
            else:
                # enable for ubo
                # light_dir[:, 0:1, :, :] = - light_dir[:, 0:1, :, :]
                # camera_dir[:, 0:1, :, :] = - camera_dir[:, 0:1, :, :]
                camera_dir[:, 1:2, :, :] = -camera_dir[:, 1:2, :, :]
                light_dir[:, 1:2, :, :] = -light_dir[:, 1:2, :, :]


        if rays_buffer is not None:

            def to_torch(x):
                x = self.to_device(torch.Tensor(x))
                x = x.unsqueeze(0)
                x = x.permute(0,3,1,2)
                return x

            camera_dir = to_torch(rays_buffer.camera_dir)[:,:2, :,:]
            light_dir = to_torch(rays_buffer.light_dir)[:,:2, :,:]
            location = to_torch(rays_buffer.camera_target_loc)[:,:2, :,:]
            radius = to_torch(rays_buffer.camera_query_radius)[:,:1,:,:]

            valid_pixels = 1



        input_info = aux_info.InputInfo()

        input_info.light_dir_x = light_dir[:, 0:1, :, :]
        input_info.light_dir_y = light_dir[:, 1:2, :, :]

        input_info.camera_dir_x = camera_dir[:, 0:1, :, :]
        input_info.camera_dir_y = camera_dir[:, 1:2, :, :]

        input_info.mipmap_level_id = 0
        #radius = radius/100
        vars = self.get_vars()

        input, mipmap_level_id = vars.generate_input(input_info, False)
        #utils.tensor.displays(torch.log2((radius*1024).clamp(1))/10)
        #radius = radius*0



        with torch.no_grad():



            import time
            start_time = time.time()



            result, eval_output = vars.evaluate(location=location, camera_dir=camera_dir, input=input, query_radius=radius,
                                                valid_pixels=valid_pixels)
            print("--- %s seconds ---" % (time.time() - start_time))
           # exit()

        #result[...] = .3
        # multiply by cosine
        result = result*utils.la4.get_3rd_axis(light_dir[:, 0:2, :, :])

        return result, eval_output



    def defered_shading(self, output_tensor,standard_rendering, get_buffer=False):
        mask = output_tensor[:, 7:8, :, :].bool()
        original_shape = mask.shape

        result, eval_output = self.brdf_eval(output_tensor)

        standard_rendering = torch.Tensor(standard_rendering).permute(2, 0, 1).unsqueeze(0)

        light_multiplier = output_tensor[:, 10:13, :, :]

        composed = (mask*result*light_multiplier).cpu() + torch.logical_not(mask).cpu()*(standard_rendering)
        #
        # utils.tensor.displays(composed)
        # exit()

        if not get_buffer:
            utils.tensor.displays(eval_output.level_id)
            utils.tensor.displays(eval_output.level_id)

            utils.tensor.displays(mask)
            # utils.tensor.displays(camera_dir, True)
            # utils.tensor.displays(light_dir, True)
            # utils.tensor.displays(location)
            # utils.tensor.displays(output_tensor)
            #utils.tensor.displays(eval_output.neural_offset)
            utils.tensor.displays(standard_rendering)
            utils.tensor.displays(result)
            utils.tensor.displays(composed)





    def render(self, timestamp=.5, get_buffer=False, frame_id=None, return_ground=False,
               action_name=None, hero_object=None, args=None, cpp_model=None):


        output_tensor = None
        aux_buffer  = None

        valid_pixels = None
        ground = None

        if frame_id is  None:
            frame_id = 0

        if self.args.def_dataset:
            def_dataset_path = os.path.join(path_config.path_dataset, self.args.def_dataset)
            get_rays_buffer = True
            # if args.cpp_model:
            #     get_rays_buffer = False

            def_dataset = dataset.dataset_reader.DatasetMulti([def_dataset_path], get_rays_buffer=get_rays_buffer)
            rays_buffer = def_dataset[frame_id]


            if get_rays_buffer:
                valid_pixels = renderer.to_torch(rays_buffer.valid_pixels, self.to_device)
                light_multiplier = renderer.to_torch(rays_buffer.light_multiplier, self.to_device)


                ground = renderer.to_torch(rays_buffer.colors, self.to_device)
            else:
                output_tensor = rays_buffer
                output_tensor = renderer.to_torch(output_tensor, self.to_device)


            if (args is not None and args.cpp_model) or cpp_model:

                output_tensor = np.concatenate([rays_buffer.camera_dir[:,:,:2],
                                           rays_buffer.light_dir[:,:,:2],
                                           rays_buffer.camera_target_loc[:,:,:2],
                                           rays_buffer.camera_query_radius, rays_buffer.camera_query_radius,
                                           rays_buffer.valid_pixels], axis=2)
                output_tensor = torch.Tensor(output_tensor)
                output_tensor = output_tensor.unsqueeze(0)


                output_tensor = self.to_device(output_tensor)


                if not cpp_model:
                    model_path = path_config.get_value("path_export", args.cpp_model)
                    cpp_model = cpputils.TorchActiveModel(model_path)

                repeat = 10
                print("cuda repeat: " + str(repeat))

                torch.cuda.synchronize()
                with utils.Timer("cuda eval"):
                    for i in range(repeat):
                        result = cpp_model.evaluate(output_tensor, 1.)
                    torch.cuda.synchronize()

                #utils.tensor.displays(result)
                eval_output = None
            else:

                result, eval_output = self.brdf_eval(rays_buffer=rays_buffer)
                #utils.tensor.displays(eval_output.neural_offset_texture+.5)

                #utils.tensor.displays(eval_output.neural_offset_texture_e0+.5)
                #utils.tensor.displays(eval_output.neural_offset_actual+.5)
                #utils.tensor.displays(result)
                #utils.tensor.displays(eval_output.neural_offset_actual + .5)
                #utils.tensor.displays(eval_output.camera_dir + .5)

                #utils.tensor.displays(eval_output.neural_texture +.5)
                #exit()



            result = result*valid_pixels*light_multiplier#*.2/5
            out_path = os.path.join(path_config.path_images, self.args.out)


            if not args.png_only and eval_output is not None:
                if eval_output.shadows_mask is not None:
                    utils.tensor.save_png(eval_output.shadows_mask, out_path+ "_shadow.png")

                if eval_output.neural_offset_actual is not None:
                    neural_offset_actual = eval_output.neural_offset_actual*4 +.5

                    neural_offset_actual = neural_offset_actual.repeat([1, 3, 1, 1])
                    neural_offset_actual = neural_offset_actual[:, :3, :, :]
                    neural_offset_actual[:, 2:, :, :] = 0

                    utils.tensor.save_png(neural_offset_actual, out_path+ "_nom.png")
                utils.tensor.save_exr(result, out_path + ".exr")

            utils.tensor.save_png(result, out_path + ".png")

            # em = error_metric.ErrorMetric(ground, result, valid_pixels, out_path, False)
            # em.save()
            print(out_path)

            #utils.tensor.displays(result)
            if False:
                utils.tensor.displays(ground)

                utils.tensor.displays(torch.abs(result - ground)*2)

            composed = result

        else:
            scene_info = renderer.setup_scene(False)

            if action_name is not None:
                scene_info.action_name = action_name

            if hero_object is not None:
                scene_info.hero_object = hero_object

            if args is not None:
                if args.zoom_level is not None:
                    scene_info.zoom_level = args.zoom_level

                if args.action_name is not None:
                    scene_info.action_name = args.action_name

                if args.spp is not None:
                    scene_info.spp = args.spp

                if args.timestamp is not None:
                    timestamp = args.timestamp

                if args.light_pan is not None:
                    scene_info.light_pan_override = args.light_pan


            scene_info.timestamp = timestamp
            standard_rendering, output_tensor = renderer.render_mitsuba(scene_info, evaluator=self)
            #utils.tensor.displays(standard_rendering)

            if True:
                out_path = os.path.join(path_config.path_images, self.args.out)

                utils.tensor.save_png(standard_rendering, out_path + ".png")
                utils.tensor.save_exr(standard_rendering, out_path + ".exr")

            if output_tensor is not None:
                output_tensor = self.to_device(output_tensor)

                composed = self.defered_shading(output_tensor, standard_rendering)
            else:
                composed = standard_rendering

        if return_ground:
            return  composed, ground, valid_pixels
        else:
            return composed
