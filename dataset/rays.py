import torch
import numpy as np
import utils
import dataset.ubo2014specs
import matplotlib.pyplot as plt


class RaysInfo:
    def __init__(self, index, length, _, ubo2014):

        self.camera_dir = None
        self.light_dir = None







class Buffers:
    def __init__(self, scene_info=None, _=False, index=None, total_length=None, current_resolution=None, lod=None):
        # Format [height, width, channels]

        self.index  = index
        self.total_length = total_length

        self.camera_origin = None
        self.camera_target_loc = None
        self.camera_dir = None
        self.camera_query_radius = None

        self.base_radius = None

        self.light_dir = None

        self.resolution = None

        self.colors = None

        self.valid_pixels = None

        self.light_multiplier = None



        if scene_info and current_resolution is None:
            current_resolution = scene_info.resolution


        self.space_index = None

        if scene_info is not None:
            self.ubo2014 = scene_info.ubo2014
            self.generate_buffer(scene_info, current_resolution, lod)

            self.resolution = scene_info.resolution


        else:
            self.ubo2014 = False


    def display(self):
       # print("Color mean", self.colors.mean())
        print("Light mult mean", self.light_multiplier.mean())

        # utils.tensor.displays(self.light_dir * .5 + .5)
        # utils.tensor.displays(self.light_dir[:,:,:2] * .5 + .5)
        #utils.tensor.displays(self.camera_dir * .5 + .5)
        # utils.tensor.displays(self.valid_pixels)
        utils.tensor.displays(self.camera_target_loc , norm=True)
        # utils.tensor.displays((self.camera_query_radius * 128))
        # utils.tensor.displays(self.light_multiplier / 30)

    def convert_to_line(buffers):
        def converter(x):
            return x.reshape(-1, x.shape[-1])

        buffers.camera_origin = converter(buffers.camera_origin)
        buffers.camera_target_loc = converter(buffers.camera_target_loc)
        buffers.camera_dir = converter(buffers.camera_dir)
        buffers.camera_query_radius = converter(buffers.camera_query_radius)
        buffers.light_dir =  converter(buffers.light_dir)



    def get_length(self):
        assert len(self.camera_dir.shape) == 2

        return self.camera_dir.shape[0]

    # heigh, width
    # angle_type, x, y, x, y, lod
    def generate_angular(self,  scene_info, current_resolution,
                         angle_type, uv_loc, other_loc, lod):

        self.camera_target_loc = np.zeros(current_resolution + [3], dtype=np.float32)
        self.camera_target_loc[:,:,0] = uv_loc[0]
        self.camera_target_loc[:,:,1] = uv_loc[1]

        uniform_dir = np.zeros(current_resolution + [2], dtype=np.float32)
        uniform_dir[:,:,0] = other_loc[0]
        uniform_dir[:,:,0] = other_loc[1]
        uniform_dir = utils.la3np.add_3rd_axis_neg(uniform_dir)

        varying_dir = utils.tensor.generate_grid(current_resolution[0], current_resolution[1])[0,:,:,:]
        varying_dir = varying_dir.permute(1,2,0).data.cpu().numpy()
        varying_dir = utils.la3np.add_3rd_axis_neg(varying_dir)

        if angle_type:
            self.camera_dir = uniform_dir
            self.light_dir = varying_dir
        else:
            self.camera_dir = varying_dir
            self.light_dir = uniform_dir

        self.calc_origin()

        radius_multiplier = np.ones(current_resolution + [1], dtype=np.float32)
        radius_multiplier = radius_multiplier * 2**lod
        radius_multiplier = radius_multiplier * calc_base_radius(scene_info.resolution[0])


        self.camera_query_radius = radius_multiplier

        self.valid_pixels = utils.la3np.get_circle(varying_dir).astype(np.float32)

        self.light_multiplier = np.ones_like(self.light_dir)

        self.finilize()


    def generate_from_pbrt(self, colors, queries):
        colors = colors.astype(dtype=np.float32)
        self.colors = colors


        self.camera_target_loc = queries[:, :, 0:2]
        self.camera_query_radius = queries[:, :, 2:3]
        self.camera_dir = -queries[:,:,3:5]
        self.light_dir = -queries[:,:,5:7]

        self.finilize()



    def generate_from_btf(self, colors, angles):
        colors = colors.astype(dtype=np.float32)
        #print(angles[:,:,3:4].min())

        angles = angles.astype(dtype=np.float32)*np.pi/180

        phiV = angles[:,:,0:1]
        thetaV = angles[:,:,1:2]
        phiL = angles[:,:,2:3]
        thetaL = angles[:,:,3:4]

        self.colors = colors

        neg_count = (self.colors <= 0).sum()
        neg_count += (~np.isfinite(self.colors)).sum()
        if neg_count > 0:
            print(neg_count)
            assert False
#

        def get_3d_coord_from_sphere(theta, phi):
            radius = np.cos(theta)
            x = - radius * np.cos(phi)
            y =  -radius * np.sin(phi)
            z =  -np.sin(theta)
            return np.concatenate([x,y,z], axis=-1)


        self.camera_dir = get_3d_coord_from_sphere(thetaV, phiV)
        self.light_dir = get_3d_coord_from_sphere(thetaL, phiL)

        self.camera_target_loc, _ = camera_generate_origin(self.scene_resolution, self.scene_resolution, True)
        self.camera_query_radius = np.zeros_like(self.light_dir[:,:,:1])

        self.finilize()





    def get_pbrt(self):
        array =  np.concatenate([self.camera_target_loc[:,:,:2],
                                self.camera_query_radius[:,:,:1],
                                self.camera_dir[:,:,:2],
                                self.light_dir[:,:,:2] ],
                                axis=-1)

        array = array.reshape(-1).astype(np.float32)

        return  array



    def generate_buffer(buffers, scene_info, current_resolution, lod):
        current_resolution = current_resolution
        buffers.scene_resolution = scene_info.resolution

        rays_info = RaysInfo(buffers.index, buffers.total_length, None, buffers.ubo2014)



        buffers.camera_origin, buffers.camera_target_loc, buffers.camera_dir, buffers.base_radius = \
            generate_camera(current_resolution, buffers.scene_resolution,
                            scene_info.fixed_camera, scene_info.fixed_patch, scene_info.smooth,
                            rays_info, scene_info.ubo2014
                            )
        buffers.camera_query_radius = camera_generate_path_size(current_resolution, scene_info.num_level, buffers.base_radius,
                                                                scene_info.fixed_patch, lod
                                                                )

        buffers.light_dir = light_generate_dir(current_resolution, scene_info.fixed_light, scene_info.smooth, rays_info, scene_info.ubo2014)
        return buffers

    def calc_resolution(self):
        self.resolution = self.camera_dir.shape[:2]

    def calc_origin(self):
        self.camera_origin = calc_origin(self.camera_target_loc, self.camera_dir)

    def fix_sizes(self):
        if self.light_dir.shape[-1] == 2:
            self.light_dir = utils.la3np.add_3rd_axis_neg(self.light_dir)

        if self.camera_dir.shape[-1] == 2:
            self.camera_dir = utils.la3np.add_3rd_axis_neg(self.camera_dir)

        if self.camera_target_loc.shape[-1] == 2:
            self.camera_target_loc = utils.la3np.add_3rd_0(self.camera_target_loc)

    def finilize(self):
        self.fix_sizes()
        self.calc_resolution()
        self.calc_origin()

def generate_dir_ubo2014(resolution):
    import dataset.ubo2014specs

    size = resolution[0]*resolution[1]

    result = np.random.choice(len(dataset.ubo2014specs.directions), size)
    directions = np.array(dataset.ubo2014specs.directions)

    result = directions[result]


    result = np.array(result)

    result =  np.reshape(result, (resolution[0], resolution[1], 3))

    result[:,:,2] = -result[:,:,2]

    # dir x is flipped
    result[:,:,0] = -result[:,:,0]




    result = result.astype(np.float32)

    result = np.ascontiguousarray(result)

    return result



def generate_dir(resolution, fixed_camera, smooth, camera, rays_info=None, ubo2014=False):
    if smooth and not fixed_camera and resolution[0]>8:
        return camera_generate_dir_smooth(resolution, camera, rays_info)

    if ubo2014:
        return generate_dir_ubo2014(resolution)

    height, width = resolution
    shape = [height, width, 1]

    if camera:
        pre_dir = rays_info.camera_dir
    else:
        pre_dir = rays_info.light_dir

    if pre_dir is not None:
        dir_x = torch.zeros(shape)
        dir_x[:,:,:] = pre_dir[0]


        dir_y = torch.zeros(shape)
        dir_y[:,:,:]  = pre_dir[1]


    else:
        theta = torch.rand(shape) * np.pi * 2

        if fixed_camera:
            radius = torch.sqrt(torch.zeros(shape))
        else:
            radius = torch.sqrt(torch.rand(shape)*.99)

        #radius = torch.pow(radius, .33) # bias to sides

        dir_x = radius * torch.sin(theta) #*0 +.4
        dir_y = radius * torch.cos(theta)#*0 +.4

        # if camera:
        #     dir_x = dir_x*0+.8
        #     dir_y = dir_y*0


    dir_z = -torch.sqrt(1 - (dir_x*dir_x + dir_y*dir_y))

    dir = torch.cat([dir_x, dir_y, dir_z], -1)

    # dir[:, :100, :2] = 0
    # dir[:, :100, 2:] = 1

    dir = dir.float()

    dir = dir.data.cpu().numpy()
    dir = np.ascontiguousarray(dir)


    return dir


def camera_generate_dir_smooth(resolution, camera, rays_info):
    start_dir = generate_dir([1,1], False, False, camera, rays_info)
    start_dir = start_dir[:, :, :2]

    height, width = resolution
    shape = [height, width, 1]

    scale = 32

    scale = min(scale, height)

    dir = np.random.normal(0, 0.2, [int(np.ceil(height/scale)), int(np.ceil(width/scale)), 2])
    dir = np.repeat(dir, scale, 0)
    dir = np.repeat(dir, scale, 1)

    dir = dir[:height, :width, :]

    import scipy

    if scale > 4:
        dir = np.stack([scipy.ndimage.filters.gaussian_filter(dir[:,:,0], scale/2),
                              scipy.ndimage.filters.gaussian_filter(dir[:, :, 1], scale/2),
                              ], -1)

    dir = dir + start_dir

    dir_length = np.sqrt(dir[:,:,0:1]*dir[:,:,0:1] + dir[:,:,1:2]*dir[:,:,1:2])

    overshoot = np.maximum(dir_length - .98, 0)
    dir = dir - dir*overshoot*2

    dir_z = -np.sqrt(np.maximum(1 - (dir[:,:,0:1]*dir[:,:,0:1] + dir[:,:,1:2]*dir[:,:,1:2]), 0))
    dir = np.concatenate([dir, dir_z], -1)

    dir = np.asarray(dir, dtype="float32")
    dir = np.ascontiguousarray(dir)
    #utils.tensor.displays(dir*.5+.5)
    return dir



def get_patch_size_linear(shape, num_resolution):
    patch_size = torch.rand(shape)*(num_resolution - 1)
    patch_size = torch.pow(2, patch_size)
    return patch_size

def get_patch_size_proportional(shape):
    y = torch.rand(shape)
    return 1/torch.sqrt(y+1e-6)


def get_patch_size_zero(shape):
    return torch.ones(shape)



def camera_generate_path_size(resolution, num_resolution, base_radius, fixed_patch, lod):

    height, width = resolution
    shape = [height, width, 1]

    if lod is not None:

        height, width = resolution
        shape = [height, width, 1]

        scale = 32

        scale = min(scale, height)

        patch_size = np.random.normal(lod, .5, [int(np.ceil(height / scale)), int(np.ceil(width / scale)), 1])
        patch_size = np.repeat(patch_size, scale, 0)
        patch_size = np.repeat(patch_size, scale, 1)

        patch_size = patch_size[:height, :width, :]

        import scipy

        if scale > 4:
            patch_size = scipy.ndimage.filters.gaussian_filter(patch_size[:, :, 0], scale / 2)[:,:,np.newaxis]


        patch_size = torch.Tensor(patch_size)

        patch_size = patch_size.clamp(0, num_resolution-1)
        patch_size = torch.pow(2, patch_size)


        #patch_size = torch.ones_like(patch_size)


    else:
        masks = utils.tensor.get_masks([.3, .6, .1], shape)
        patch_size = masks[0]*get_patch_size_linear(shape, num_resolution) + \
            masks[1]*get_patch_size_proportional(shape) + \
            masks[2] * get_patch_size_zero(shape)

    if fixed_patch:
        patch_size = torch.ones_like(patch_size)


    patch_size = patch_size * base_radius



    patch_size = patch_size.data.cpu().numpy()
    patch_size = np.ascontiguousarray(patch_size)

    return patch_size


def light_generate_dir(resolution, fixed_light, smooth, rays_info, ubo2014):
    return generate_dir(resolution, fixed_light, smooth, camera=False, rays_info=rays_info, ubo2014=ubo2014)


def calc_base_radius(scene_height):
    return 1/scene_height / 2

def camera_generate_origin(current_resolution, scene_resolution, fixed_patch):
    height, width = current_resolution


    half_hor = 1./width
    half_ver = 1./height

    loc_x = torch.linspace(-1 + half_hor, 1 - half_hor, width)#*0+.1
    loc_y = torch.linspace(-1 + half_ver, 1 - half_ver, height)

    loc_x, loc_y = torch.meshgrid(loc_x, loc_y)


    loc_z = torch.zeros_like(loc_x)

    loc = torch.stack([loc_y, loc_x, loc_z], -1) #

    shape = [height, width, 2]

    if fixed_patch:
        pahch_size = 0
    else:
        pahch_size = 2/width

    offset =  (torch.rand(shape) - .5) *pahch_size
    loc[:,:,:2] = loc[:,:,:2] + offset

    loc = loc.float()
    loc = loc.data.cpu().numpy()

    sr_width, sr_height = scene_resolution
    assert sr_width == sr_height

    base_radius = calc_base_radius(sr_height)



    return loc, base_radius

def calc_origin(target_loc, dir):
    return target_loc - dir / np.absolute(dir[:, :, 2:]) * 10.0

def generate_camera(current_resolution, scene_resolution, fixed_camera, fixed_patch, smooth, rays_info, ubo2014):
    dir = generate_dir(current_resolution, fixed_camera, smooth, camera=True, rays_info=rays_info, ubo2014=ubo2014)
    target_loc, base_radius = camera_generate_origin(current_resolution, scene_resolution, fixed_patch)

    origin = calc_origin(target_loc, dir)


    return origin, target_loc, dir, base_radius