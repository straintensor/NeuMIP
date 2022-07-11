import dataset.sceneinfo
import dataset.generator_mitsuba
import utils
import torch


def setup_teaser(deferred):
    scene_info = dataset.generator_mitsuba.SceneInfo()
    scene_info.resolution = (720, 1280)

    #scene_info.resolution = (512, 512)

    if False:
        scene_info.material_name = "white"
        scene_info.integrator_type = "path"
    elif False:

        scene_info.material_name = "mlpmat"
        scene_info.material_path = "test1.model"
        scene_info.integrator_type = "path"


    else:
        scene_info.material_name = "neuralmat"
        scene_info.integrator_type = "wnpath"
        #scene_info.integrator_type = "wnpath"




    scene_info.scene_setup = 0

    scene_info.spp =12#64# 64#64#1#256#32#16*4
    #scene_info.spp =64


    scene_info.depth = 3
    scene_info.depth = 3

    scene_info.object_name = "knob1"

    scene_info.scene_name = "Teaser1"
    scene_info.hero_object = "cloth1"
    # scene_info.hero_object = "chest1"

    #scene_info.env_light = "Uniform1"



    #scene_info.action_name = "camera_pan"
   # scene_info.action_name = "camera_zoom"

    #scene_info.action_name = "closeup_camera_pan"
    #scene_info.action_name = None
    #scene_info.timestamp = 0


    if False:
        scene_info.env_light = None
        #scene_info.env_light = "Uniform1"
        scene_info.scene_name = "Test1"

        scene_info.hero_object = "SimplePlane"
        scene_info.lc_name = "s1" # straight camera and straight light
        scene_info.lc_name = "s4" #  45 camera and straight light


        scene_info.lc_name = "sp7" #  45 camera and straight light

        scene_info.resolution = (512, 512)

    return scene_info

def setup_test1(deferred, resolution):
    scene_info = dataset.generator_mitsuba.SceneInfo()
    scene_info.resolution = resolution



    if deferred:
        scene_info.material_name = "neuralmat"
        scene_info.integrator_type = "dnpath"
    else:
        scene_info.material_name = "grid128"
        scene_info.material_name = "white"

        scene_info.material_name = "grid128g1"

        scene_info.integrator_type = "path"

    #scene_info.integrator_type = "wnpath"
    scene_info.scene_setup = 0

    scene_info.spp = 1

    scene_info.depth = 2

    scene_info.object_name = "knob1"

    scene_info.scene_name = "Test1"

    scene_info.hero_object = "SimplePlane"
    # scene_info.hero_object = "Hill"
    # scene_info.hero_object = "Row"
    #scene_info.hero_object = "Sloppy"


    scene_info.lc_name = "s1" # straight camera and straight light
    scene_info.lc_name = "s2" #  45 camera and straight light
    #scene_info.lc_name = "s3" #  straight camera and 45 light
    scene_info.lc_name = "s4"  # straight camera and 45 light

    #scene_info.lc_name = "s5" #  straight camera and 30 light


    scene_info.env_light = None

    return scene_info

def setup_scene(deferred, timestamp=0.5):
   # return setup_test1(deferred)
    return setup_teaser(deferred)

    scene_info = dataset.generator_mitsuba.SceneInfo()
    scene_info.resolution = (512, 512)
    #scene_info.resolution = (1024, 1024)

    if True or deferred:
        scene_info.material_name = "neuralmat"
        #scene_info.material_name = "grid128d"
    else:
        scene_info.material_name = "rocks_ground_05"

    if True:
        scene_info.object_name = "knob1"

        scene_info.scene_setup = 0
        scene_info.action_type = 2.
    else:
        scene_info.object_name = "plane1"
        # scene_info.object_name = "rocks_ground_05"
        scene_info.scene_setup = 2

    scene_info.timestamp = timestamp
    scene_info.spp = 32
    # self.vars.use_offset = False # TODO remove this

    scene_info.integrator_type = "nrpath"
    scene_info.integrator_type = "wnpath"

    if deferred:
        scene_info.integrator_type = "dnpath"

    return scene_info

def render_mitsuba(scene_info: dataset.generator_mitsuba.SceneInfo, mr=None, evaluator=None):

    if mr is None:
        mr = dataset.generator_mitsuba.MitsubaRenderer()

    scene_data, output_tensor = mr.create_nn_scene(scene_info, None, scene_info.scene_setup, evaluator)
    standard_rendering = mr.render(scene_data)

    if output_tensor is None:
        #utils.tensor.displays(standard_rendering)
        return standard_rendering, None

    output_tensor = torch.Tensor(output_tensor).unsqueeze(0)
    output_tensor = output_tensor.permute(0, 3, 1, 2)

    return standard_rendering, output_tensor


def convert_to_rays_buffer(output_tensor, zoom_in_multiplier=1):
    camera_dir = output_tensor[:, 0:2, :, :]
    light_dir = output_tensor[:, 2:4, :, :]
    location = output_tensor[:, 4:6, :, :]



    # camera_dir = utils.la4.vector_clamp(camera_dir, .85)
    # light_dir = utils.la4.vector_clamp(light_dir, .85)

    valid_pixels = output_tensor[:, 7:8, :, :]

    radius_du = output_tensor[:, 8:9, :, :]
    radius_dv = output_tensor[:, 9:10, :, :]

    radius = ((radius_du + radius_dv) / 2.0)

    light_multiplier = output_tensor[:, 10:13, :, :]

    radius = radius*zoom_in_multiplier
    location = torch.fmod(location*zoom_in_multiplier,1)

    camera_dir = utils.la4.add_3rd_axis_neg(camera_dir)
    light_dir = utils.la4.add_3rd_axis_neg(light_dir)

    def convert_to_hwc(x: torch.Tensor):
        x = x.data.cpu().numpy()
        x = x.transpose(0,2,3,1)
        x = x[0,...]
        return x

    buffer = dataset.rays.Buffers()

    buffer.camera_dir = convert_to_hwc(camera_dir)
    buffer.light_dir = convert_to_hwc(light_dir)
    buffer.camera_target_loc = convert_to_hwc(location)*2-1



    #
    # buffer.camera_target_loc[:,:,0] = -buffer.camera_target_loc[:,::1,0]
    # buffer.camera_dir[:,:,0] = -buffer.camera_dir[:,::1,0]
    # buffer.light_dir[:,:,0] = -buffer.light_dir[:,::1,0]

    buffer.camera_target_loc[:, :, 1] = -buffer.camera_target_loc[:, ::1, 1]
    #
    buffer.camera_dir = -buffer.camera_dir
    buffer.light_dir = -buffer.light_dir

    buffer.camera_query_radius = convert_to_hwc(radius)
    buffer.valid_pixels = convert_to_hwc(valid_pixels)

    buffer.light_multiplier = convert_to_hwc(light_multiplier)

    buffer.finilize()

    return buffer


def to_torch(x, to_device):
    x = to_device(torch.Tensor(x))
    x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    return x