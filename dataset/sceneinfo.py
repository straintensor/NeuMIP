class SceneInfo:
    def __init__(self):
        self.resolution = None
        self.num_level  = None

        self.fixed_camera = False
        self.fixed_light = False
        self.fixed_patch = False

        self.object_name = None
        self.material_name = None

        self.scene_setup = 0
        self.timestamp = .5

        self.action_type = 0
        self.action_name = None

        self.tilesize = 1

        self.preloeaded_obj = None

        self.smooth = False

        self.integrator_type = None

        self.s_light_2 = False

        self.zoom_in_multiplier = 1

        self.scene_name = "simple"

        self.spp = 1
        self.depth = 3
        self.env_light = None

        self.hero_object = None

        self.lc_name = None

        self.zoom_level = None
        self.light_pan_override = None

        self.material_path = None
