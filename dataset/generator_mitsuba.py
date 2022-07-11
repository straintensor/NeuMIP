import os, sys
import numpy as np
import multiprocessing
import utils

import path_config
import scipy.ndimage.filters

from dataset.sceneinfo import SceneInfo

import dataset.rays

if True:
    if "mitsuba-build-debug" in os.environ.get("PYTHONPATH", ""):
        use_debug = True
        mitsuba_path = os.path.join(path_config.path_base, "mitsuba-build-debug/binaries")
    else:
        use_debug = False
        if not "mitsuba-build" in os.environ.get("PYTHONPATH", ""):
            os.environ["PYTHONPATH"] += os.pathsep + os.path.join(path_config.path_base, "mitsuba-build/binaries/python")

        mitsuba_path = os.path.join(path_config.path_base, "mitsuba-build/binaries")

    sys.path.append(mitsuba_path + "/python")
    os.environ['PATH'] = mitsuba_path + os.pathsep +  mitsuba_path + "/plugins" + os.pathsep +  os.environ['PATH']


    print('PATH', os.environ['PATH'] )
    print('PYTHONPATH', os.environ['PYTHONPATH'] )


try:
    import mitsuba.core as mitsuba
    import mitsuba.render as mitsuba_render
except ModuleNotFoundError as e:
    print("Skipping importing mitsuba")



import dataset.generator_mitsuba_assets
import dataset.ubo2014specs

first_run = True

def translate(x, y, z):
    return mitsuba.Transform.translate(mitsuba.Vector(x, y, z))

def scale(s, s2=None, s3=None):
    if s2 is None:
        s2 = s
    if s3 is None:
        s3 = s
    return mitsuba.Transform.scale(mitsuba.Vector(s, s2, s3))

def rot_hor(degree):
    return mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), degree)





def rot_hor3(degree):
    return mitsuba.Transform.rotate(mitsuba.Vector(0, 0, 1), degree)

def rot_ver3(degree):
    return mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), -degree)

def flip_x():
    return mitsuba.Transform.rotate(mitsuba.Vector(1, 0, 0), -90)
    return mitsuba.Transform.rotate(mitsuba.Vector(1, 0, 0), -90)

def roughconductor(spec, alpha):
    return  {
        "specularReflectance": mitsuba.Spectrum([*spec]),
        "type": "roughconductor",
        "distribution": "ggx",
        "material": "Ag",
        "alpha": alpha
    }

class MitsubaRenderer:
    def __init__(self):
        self.pmgr = mitsuba.PluginManager.getInstance()
        self.scheduler = mitsuba.Scheduler.getInstance()

        global first_run

        if first_run:
            first_run = False
            if not use_debug:
                for i in range(0, multiprocessing.cpu_count()):
                    self.scheduler.registerWorker(mitsuba.LocalWorker(i, 'wrk%i' % i))
            else:
                self.scheduler.registerWorker(mitsuba.LocalWorker(0, 'wrk%i' % 0))
            self.scheduler.start()




    def create_nn_scene(self, scene_info: SceneInfo, neural_evaluator, scene_type, evaluator=None):
        light_dir_x = 0.0
        light_dir_y = 0.0

        light_dir_z = np.sqrt(1 - (light_dir_x * light_dir_x + light_dir_y * light_dir_y))
        light_dir = mitsuba.Vector(light_dir_x, light_dir_y, light_dir_z)
        light_type = "directional"

        #simple_only = True
        simple_only = False

        use_vase = False

#scene_info.material_name)#
        #material_value = dataset.generator_mitsuba_assets.get_material("neuralmat")
        material_value = dataset.generator_mitsuba_assets.get_material(scene_info.material_name)
        if scene_info.material_name == "mlpmat":
            material_value["path"] = path_config.get_value("path_export", scene_info.material_path)
        elif scene_info.material_name == "ubo2014":
            material_value["ubo_path"] = path_config.path_ubo2014 + "/" +scene_info.material_path
            material_value["zoom_in_multiplier"] = scene_info.zoom_in_multiplier
            material_value["flip_x_dir"] = False

        object_type, object_value = dataset.generator_mitsuba_assets.get_object(scene_info.object_name, material_value)

        out_num_ch = 13


        path_assets = path_config.path_assets + "/"

        light_intensity = 10.

        rfilter = {
                             "type": "box",
                            "radius": 0.5,
                         }
        if scene_info.integrator_type is None or scene_info.integrator_type == "dnpath":

            output_tensor = np.zeros((list(scene_info.resolution) + [out_num_ch]), "float32")
            integrator = {
                    "type": "dnpath",
                    "output_tensor": mitsuba.TensorFloat3(output_tensor),
                    "maxDepth": 2,
                    "center_only": True,


                }
            sampler =  {
                'type': 'independent',
                'sampleCount': 1
            }

        elif scene_info.integrator_type == "nrpath":
            output_tensor = None
            integrator = {
                "type": "nrpath",
                "maxDepth": 2,
                "center_only": False,
                "evaluator": evaluator
            }
        elif scene_info.integrator_type == "wnpath":
            output_tensor = None
            integrator = {
                "type": "wnpath",
                "maxDepth": scene_info.depth,
                "center_only": False,
                "evaluator": evaluator,
                "misType": 0,
                #"use_async_eval": True
            }
            sampler = {
                #'type': 'ldsampler',
                'type': 'independent',
               # 'type': 'indsimple',
                'sampleCount': scene_info.spp
            }
            rfilter = {"type": "gaussian" }

        elif scene_info.integrator_type == "path":
            output_tensor = None
            integrator = {
                "type": "path",
                "maxDepth": scene_info.depth,
                "center_only": False,
                #"evaluator": evaluator,
                "misType": 0,
                #"use_async_eval": True
            }
            sampler = {
                #'type': 'ldsampler',
                'type': 'independent',
               # 'type': 'indsimple',
                'sampleCount': scene_info.spp
            }
            rfilter = {"type": "gaussian" }



        upvector = mitsuba.Vector(0, .1, 1)


        def crane(origin, target, zoom=1., pan=.0, tilt=0.):
            target = mitsuba.Vector(target)
            origin = mitsuba.Vector(origin)

            diff = (origin - target )*zoom

            diff = mitsuba.Transform.rotate(mitsuba.Vector(1, 0, 0), tilt)*diff
            diff = mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), pan)*diff

            origin = target + diff

            return mitsuba.Point(origin)


        def camera_crame(origin, target, **kargs):
            origin = crane(origin, target, **kargs)

            target = mitsuba.Point(target)
            view = mitsuba.Transform.lookAt((origin),
                                            (target),
                                            upvector)
            return view


        def crane2(origin, target, zoom=1., pan=.0, tilt=0.):
            target = mitsuba.Vector(target)
            origin = mitsuba.Vector(origin)

            diff = (origin - target )*zoom

            diff = mitsuba.Transform.rotate(mitsuba.Vector(1, 0, 0), tilt)*diff
            diff = mitsuba.Transform.rotate(mitsuba.Vector(0, 0, 1), pan)*diff

            origin = target + diff

            return mitsuba.Point(origin)


        def camera_crame2(origin, target, **kargs):
            origin = crane2(origin, target, **kargs)

            target = mitsuba.Point(target)
            view = mitsuba.Transform.lookAt((origin),
                                            (target),
                                            upvector)
            return view



        if scene_type == 0:

            def get_light_pos(time):
                light_pos = (0, 3., 3.5)
                light_pos = mitsuba.Point(*light_pos)

                if scene_info.action_type == 3:
                    light_pos = mitsuba.Transform.rotate(mitsuba.Vector(1, 0, 0), (time - .5) * 60) * light_pos
                else:
                    light_pos = mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), (time - .5) * 60) * light_pos
                light_pos = mitsuba.Point(light_pos)
                return light_pos

            def get_camera_view(time):
                target =  mitsuba.Vector(0, .7, 0)
                origin = mitsuba.Vector(0, 2, 5)
                dir = origin - target
                if scene_info.action_type == 1:
                    origin = dir * (1 + time) + target
                elif scene_info.action_type == 2:
                    origin = mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), (time - .5)*60) * dir
                    origin = (origin + mitsuba.Vector(target))

                view = mitsuba.Transform.lookAt(mitsuba.Point(origin),
                                         mitsuba.Point(target),
                                         mitsuba.Vector(0, 1, 0))

                return view

            light_pos = get_light_pos(.5)
            camera_view = get_camera_view(0)


            if scene_info.action_type == 0 or scene_info.action_type == 3:
                light_pos = get_light_pos(scene_info.timestamp)
            elif scene_info.action_type == 1 or scene_info.action_type == 2:

                camera_view = get_camera_view(scene_info.timestamp)





        elif scene_type == 1 or scene_type == 2:
            if scene_type == 1 :
                light_pos = mitsuba.Point(0., 0., -4.)
                light_color = mitsuba.Spectrum([100.0, 100.0, 100.0])
            elif scene_type == 2:
                light_pos = mitsuba.Point(0., 0., -2.)
                light_color = mitsuba.Spectrum([10.0, 10.0, 10.0])


        def get_time(action_name, default=.5):
            if isinstance(action_name, list):
                if scene_info.action_name in action_name:
                    return scene_info.timestamp
            elif action_name == scene_info.action_name:
                return scene_info.timestamp
            return default

        camera_pan_time = get_time( "camera_pan", 0.5)

        closeup_camera_pan_time = get_time("closeup_camera_pan", 0.5)
        camera_tilt_time = get_time("camera_tilt", 0)
        camera_zoom_time = get_time("camera_zoom",0)

        light_pan_time = get_time("light_pan",0)
        if scene_info.light_pan_override is not None:

            light_pan_time = scene_info.light_pan_override


        light_tilt_time = get_time("light_tilt", 0)

        def lerp(a, b, t):
            return a*(1-t) + b*t


        if scene_info.scene_name == "Teaser1":
            if scene_info.hero_object == "cloth1":
                target = mitsuba.Vector(0, 0, .2)

                zoom_in_position = .26

                if scene_info.zoom_level is not None:
                    zoom_in_position = lerp(.26, 1, scene_info.zoom_level)

                zoom = lerp(1, zoom_in_position, camera_zoom_time)
            elif scene_info.hero_object == "chest1":
                target = mitsuba.Vector(0, -0.2, .55)
                zoom = lerp(1,.17, camera_zoom_time)

            target = target
            origin = mitsuba.Vector(0, 0, 5) + target

            pan = lerp(-50, 10, camera_pan_time)
            closeup_pan = lerp(-45, 45, closeup_camera_pan_time)

            if scene_info.action_name == "closeup_camera_pan":
                origin = crane2(origin,
                                       target, zoom=1,
                                        pan=pan,
                                       tilt=60#;lerp(-20, 20, camera_tilt_time),
                                       )
                origin = mitsuba.Vector(origin)

                new_target = lerp(target, origin, .19)
                new_origin = lerp(target, origin, zoom_in_position) #zoom_in_position

                origin = new_origin
                target = new_target


                camera_view = camera_crame2(origin,
                              target, zoom=1,
                              pan=closeup_pan,
                              tilt=0  # ;lerp(-20, 20, camera_tilt_time),
                              )

            else:



                camera_view = camera_crame2(origin,
                                           target, zoom=zoom,
                                            pan=pan,
                                           tilt=60#;lerp(-20, 20, camera_tilt_time),
                                           )

            #print(camera_view)

            camera_origin = camera_view*mitsuba.Vector4(0,0,0,1)
            camera_origin = mitsuba.Vector(camera_origin[0], camera_origin[1], camera_origin[2])

            #print(camera_origin)
            #camera_origin = *
            #exit()

        if scene_info.scene_name == "Test1":
            target = mitsuba.Vector(0, 0, 0)
            zoom = 1

            tilt =0

            if scene_info.lc_name in ["s2","s4"]:
                tilt = 45# + 20
            if scene_info.lc_name in ["s1", "s3", "s5"]:
                tilt = 0


            if scene_info.zoom_level is not None:
                zoom_level = scene_info.zoom_level
                #zoom_level = zoom_level*zoom_level
                zoom_level = .5 - .5 * np.cos(zoom_level *np.pi)

                zoom = lerp(2, 20,zoom_level)


            if scene_info.lc_name not in ["sp7"]:
                camera_view = camera_crame(mitsuba.Vector(0, 0, 2.3) + target,
                                           target, zoom=zoom,
                                           pan=0,
                                           tilt=tilt,
                                           )
            else:


                origin = dataset.ubo2014specs.directions[40]

                camera_view = mitsuba.Transform.lookAt( ( mitsuba.Point(*origin)) ,
                                                           ( mitsuba.Point(0, 0, 0)),

                                                upvector)

            camera_view_m = camera_view.getMatrix()

            import sys
            #matrix_file = sys.stdout
            matrix_file = open("/home/alex/projects/neural-mipmap/datasets/animation4/matrices.txt", "a")

            if False:
                matrix_file.write("matrices.append(")
                matrix_file.write("[")
                for j in range(4):
                    matrix_file.write("[")
                    for i in range(4):
                        matrix_file.write(str( camera_view_m[j,i]) + ", ")

                    matrix_file.write("],\n")

                matrix_file.write("]")
                matrix_file.write(")\n")
            else:
                matrix_file.write("cameras_pos.append([" + str(camera_view_m[0,3]) + "," + \
                                  str(camera_view_m[1,3]) + ","+\
                                    str(camera_view_m[2,3]) +\
                                  "])\n")

            matrix_file.flush()
            #exit()



        camera_type = 'perspective'
        if scene_info.lc_name in ["sp7"]:
            camera_type = 'orthographic'

        scene = {
            'type': 'scene',

            "integrator": integrator,

            "camera": {
                'type': camera_type,
                "fov": 45.,
                'toWorld': camera_view,
                #"fovAxis": "diagonal",
                # mitsuba.Transform.translate(mitsuba.Vector(0, 0, -5)),

                'sampler': sampler,

                'film': {
                    'type': 'hdrfilm',
                    'width': scene_info.resolution[1],
                    'height': scene_info.resolution[0],
                    "gamma": 1.,
                    "rfilter": rfilter
                },
            }

        }

        scene_dict = scene

        scene = self.pmgr.create(scene)




        def add_obj(x):
            scene.addChild(self.pmgr.create(x))




        if scene_type == 0 and scene_info.scene_name is  None:
            # add_obj( )
            if True:
                add_obj(
                    {
                        'type': 'rectangle',
                        'toWorld': flip_x() *
                                   scale(20.),
                        "bsdf": {
                            # "reflectance": mitsuba.Spectrum([0.5, 1.0, 0.5]),
                            # "type": "diffuse",
                            "specularReflectance": mitsuba.Spectrum([0.5, 1.0, 0.5]),
                            "type": "roughconductor",
                            "distribution": "ggx",
                            "material": "Ag",
                            "alpha": 0.13
                        }})

                size = 1.2
                add_obj(
                    {
                        'type': 'sphere',
                        'toWorld':
                            mitsuba.Transform.rotate(mitsuba.Vector(0, 1, 0), -10.) *
                            translate(2.1, 1.5, 0.) *
                            scale(size),
                        "bsdf": {
                            "specularReflectance": mitsuba.Spectrum([1.0, 1.0, 0.5]),
                            "type": "roughconductor",
                            "distribution": "ggx",
                            "material": "Ag",
                            "alpha": 0.1
                        }})



            add_obj({
                "type": "point",
                "position": light_pos,  # (0, 0, 1)
                "intensity": mitsuba.Spectrum([light_intensity, light_intensity, light_intensity])

            })

        elif scene_type == 1 or scene_type == 2:

            add_obj({
                    'type': 'perspective',
                    "fov": 45.,
                    'toWorld':  # mitsuba.Transform.translate(0, 0, -5),
                        mitsuba.Transform.lookAt(mitsuba.Point(0, 0, -2.4),
                                                 mitsuba.Point(0, 0, 0),
                                                 mitsuba.Vector(0, -1, 0)),
                    'sampler': sampler,

                    'film': {
                        'type': 'hdrfilm',
                        'width': scene_info.resolution[1],
                        'height': scene_info.resolution[0],
                        "gamma": 1.,
                        "rfilter": rfilter

                    }})

            add_obj( {
                    "type": "point",
                    "position": light_pos,  # (0, 0, 1)
                    "intensity": light_color

                })

        # add_obj(object_value)

        if scene_info.s_light_2:
            add_obj({
                "type": "point",
                "position": mitsuba.Point(-3., 3., 3.5),  # (0, 0, 1)
                "intensity": mitsuba.Spectrum([light_intensity, light_intensity, light_intensity])

            })

        if scene_info.env_light is not None and not simple_only:
            if scene_info.env_light == "Uniform1" and True:
                add_obj({"type":"constant",
                         #"samplingWeight": 1.,#.1,
                         "radiance": mitsuba.Spectrum(0.3)
                         })

        #
        # add_obj({"type":"constant",
        #          #"samplingWeight": 1.,#.1,
        #          "radiance": mitsuba.Spectrum(0.3)
        #          })

        if scene_info.scene_name == "Test1":
            base_path = path_assets + "Setup1/meshes/scene/Scene/00001/"

            heros = {
                "SimplePlane": "Plane_0000_m000.serialized",
                "Hill": "Plane_001_0000_m000.serialized",
                "Row": "Plane_002_0000_m000.serialized",
                "Sloppy": "Plane_003_0000_m000.serialized",
            }

            hero_name =heros[scene_info.hero_object]

            object_value = {
                "type": "serialized",
                "filename": base_path + hero_name,
                "bsdf": material_value}


            if False:
                add_obj(object_value )
            else:
                tilesize = 1
                tilesize = 20
                shapegroup = {
                    "type": "shapegroup",
                    "id": "MainObject",
                    "shape": object_value
                }

                shapegroup = self.pmgr.create(shapegroup)
                scene.addChild(shapegroup)
                for y_off in range(-tilesize, tilesize+1, 1):
                    for x_off in range(-tilesize, tilesize+1, 1):
                        c_object_value = dict(object_value)

                        trans = mitsuba.Transform.translate(mitsuba.Vector(2*x_off, 2*y_off, 0))
                        # if "toWorld" in c_object_value:
                        #     c_object_value["toWorld"]=trans*c_object_value["toWorld"]
                        # else:
                        #     c_object_value["toWorld"]=trans
                        # x = self.pmgr.create(c_object_value)
                        # scene.addChild(x)
                        instance = {
                            "type": "instance",
                            "shapegroup": shapegroup,
                            "toWorld": trans
                        }

                        instance = self.pmgr.create(instance)
                        scene.addChild(instance)

            l_pan = None

            if scene_info.lc_name in ["s1", "s2"] or True:
                l_tilt = 0

                l_pan = 0
                distance = 2
            if scene_info.lc_name in ["s3", "s4"]:
                l_tilt = 45
                l_pan = 30
                distance = 10
            if scene_info.lc_name in ["s5"]:
                l_tilt = 40
                l_pan = 0
                distance = 10

            # light = {
            #
            #     'toWorld':
            #         rot_hor3(l_pan) *
            #         rot_ver3(l_tilt) *
            #         translate(0, 0, distance) ,
            #
            #     "type": "point",
            #     "intensity": mitsuba.Spectrum(4  * distance * distance)
            #
            # }

            if l_pan is not None:
                l_dir = rot_hor3(l_pan) *\
                        rot_ver3(l_tilt) *\
                        translate(0, 0, distance)

                l_dir = -(l_dir*mitsuba.Vector4(0,0,0,1))

            if scene_info.lc_name in ["sp7"]:
                l_dir = dataset.ubo2014specs.directions[50]
                l_dir = [-x for x in l_dir]

            #l_dir = [0,0, -1]
            light = {

                'direction': mitsuba.Vector(l_dir[0], l_dir[1], l_dir[2]),

                "type": "directional",
                "irradiance": mitsuba.Spectrum(4)

            }

            add_obj(light)

            if True:
                print("camera_toWorld = ", camera_view)
                print("fov = ", scene_dict["camera"]["fov"])
               # print("light_pos = ", light["toWorld"]*mitsuba.Vector4(0, 0, 0, 1))
                print("light_direction = ", light["direction"])
                print("light_irradiance = ", light["irradiance"])

                #exit()


        if scene_info.scene_name == "Teaser1":


            add_obj(
                {
                    'type': 'rectangle',
                    'toWorld':  scale(20),
                    "bsdf": {
                        "reflectance": {
                            "type": "bitmap",
                            "name": "reflectance",
                            "warpMode": "repeat",
                            "filename": path_assets + "Teaser1/plaster_grey_04_diff_4k.png",
                            "vscale": 20./5,
                            "uscale": 20./5.,
                        },
                        "type": "diffuse",
                    }})


            if scene_info.hero_object == "cloth1":

                #print(self.pmgr.getLoadedPlugins())
                add_obj({
                    "type": "serialized",
                    "filename": path_assets + "Teaser1/cloth2.serialized",
                    'toWorld':   translate(0, 0,-.03),
                    "bsdf": material_value,
                    "maxSmoothAngle": 30.

                })
                print(material_value)
                print(self.pmgr.getLoadedPlugins())

                #exit()
            elif scene_info.hero_object == "chest1":
                transform =  translate(0, 0,-.03) * scale(2)
                add_obj({
                    "type": "serialized",
                    "filename": path_assets + "Teaser1/chest1.serialized",
                    'toWorld': transform,
                    "bsdf": roughconductor([.2, .15, .1], 0.5)
                })
                add_obj({
                    "type": "serialized",
                    "filename": path_assets + "Teaser1/chest2.serialized",
                    'toWorld': transform,
                    "bsdf": material_value
                })

            if False:
                add_obj({
                    "type": "point",
                    "position": mitsuba.Point(-3., 3., 3.5),  # (0, 0, 1)
                    "intensity": mitsuba.Spectrum(100)

                })
            if  False:
                add_obj({
                    "type": "directional",
                    "direction": mitsuba.Vector(-3., -3., -3.5),  # (0, 0, 1)
                    "irradiance": mitsuba.Spectrum(3)

                })
            if True:

                size = 0.4
                distance = 100
                size *=distance
                light_radiance = mitsuba.Spectrum(2/size/size*distance*distance)
                light_toWorld = rot_hor3(lerp(180, 270, light_pan_time)) *\
                        rot_ver3(lerp(30, 40, light_tilt_time))*\
                        translate(distance, 0, 0.) *\
                        scale(size)


                if not simple_only:
                    add_obj({
                        "type": "sphere",
                        'toWorld': light_toWorld,
                        "emitter": {
                            "type": "area",
                            "radiance": light_radiance
                        }
                    })
                else:
                    add_obj({
                        "type": "point",
                        'position': mitsuba.Point(camera_origin),
                        "intensity": light_radiance
                    })

            if False:
                size = 1.2
                size = 0.7
                add_obj(
                    {
                        'type': 'sphere',
                        'toWorld':
                            rot_hor( -10.) *
                            translate(2.1, size, 0.) *
                            scale(size),
                        "bsdf": {
                            "specularReflectance": mitsuba.Spectrum([.7, .9, .7]),
                            "type": "roughconductor",
                            "distribution": "ggx",
                            "material": "Ag",
                            "alpha": 0.05
                        }})
            else:
                if use_vase:
                    add_obj({
                        "type": "serialized",
                        "filename": path_assets + "Teaser1/vase.serialized",
                        'toWorld':
                            rot_hor3(30.) *
                            translate(2.4, 0, 0.) *
                                   translate(0, 0, -.03) *\
                                    scale(.7,.7,.5),
                        "bsdf": roughconductor([.4, .9, .7], 0.13) # 0.05
                    })



            if False:
                size = 1.4
                add_obj(
                    {
                        'type': 'sphere',
                        'toWorld':
                            rot_hor(110.) *
                            translate(2.5, size, 0.) *
                            scale(size),
                        "bsdf": {
                            "specularReflectance": mitsuba.Spectrum([1.0, .6, .6]),
                            "type": "roughconductor",
                            "distribution": "ggx",
                            "material": "Ag",
                            "alpha": 0.01
                        }})
            else:
                if use_vase:
                    add_obj({
                        "type": "serialized",
                        "filename": path_assets + "Teaser1/vase2.serialized",
                        'toWorld':
                            rot_hor3(110.) *
                            translate(2.5, 0, 0.) *
                                   translate(0, 0, -.03) *\
                                    scale(1),
                        "bsdf": roughconductor([1.0, .6, .6], 0.13)
                    })

        #add_obj({"type":"sphere"})

        sceneResID = self.scheduler.registerResource(scene)
        return (scene, sceneResID), output_tensor





    def create_scene(self, scene_info: SceneInfo, light_dir_x=None, light_dir_y=None, buffers=None):

        # logger = mitsuba.Thread.getThread().getLogger()
        # logger.setLogLevel(mitsuba.EError)
        if buffers is None:
            buffers = dataset.rays.Buffers(scene_info)


        camera_origin = mitsuba.TensorFloat3(buffers.camera_origin)
        camera_dir = mitsuba.TensorFloat3(buffers.camera_dir)
        buffers.camera_query_radius = buffers.camera_query_radius# *0+ buffers.base_radius*32
        camera_query_radius2 = mitsuba.TensorFloat3(buffers.camera_query_radius * 2.0) # multiply by 2 because image is from -1 to 1


        if light_dir_x is None:
            del light_dir_x
            del light_dir_y
            light_dir = mitsuba.TensorFloat3(buffers.light_dir)

            light_type = "customdirectional"

        else:

            light_dir_z = np.sqrt(1 - (light_dir_x*light_dir_x + light_dir_y *light_dir_y))

            light_dir = mitsuba.Vector(light_dir_x, light_dir_y, light_dir_z)

            light_type = "directional"

        if scene_info.preloeaded_obj is None:
            material_value = dataset.generator_mitsuba_assets.get_material(scene_info.material_name)
            object_type, object_value = dataset.generator_mitsuba_assets.get_object(scene_info.object_name, material_value)
            #preloeaded_obj = self.pmgr.create(object_value)

        else:
            preloeaded_obj = scene_info.preloeaded_obj




        scene = self.pmgr.create({
            'type' : 'scene',

            "integrator": {
                "type": "path",
                "maxDepth": 4,
                "center_only": False
            },

            'emitter': {
                "type": light_type,
                "direction": light_dir, #(0, 0, 1)
                "irradiance": mitsuba.Spectrum(1.0)

            },

            # 'envmap' : {
            #     'type' : 'sunsky'
            # },
            'sensor' : {
                # 'type' : 'perspective',
                #
                # "fov": 45.,
                # 'toWorld':
                #     mitsuba.Transform.lookAt(mitsuba.Point(0, 0, -2.4*3),
                #                              mitsuba.Point(0, 0, 0),
                #                              mitsuba.Vector(0, -1, 0)),

                'type': 'customcamera',

                "rayOrigin": camera_origin,
                "rayDir": camera_dir,
                "rayPatchSize": camera_query_radius2,

                #'toWorld' : mitsuba.Transform.translate(mitsuba.Vector(0, 0, -5)),
                'sampler' : {
                    'type' : 'independent',
                    'sampleCount' : 64
                },

                'film': {
                    'type': 'hdrfilm',
                     'width': buffers.resolution[1],
                    'height': buffers.resolution[0],
                     "gamma": 1.,
                     "rfilter": {
                         "type": "box",
                        "radius": 0.5,
                     }

                    },
            }
        })



        if scene_info.tilesize == 1:

            x = self.pmgr.create(object_value)
            #x = x.getProperties()

            #x = self.pmgr.createObject(x)

            scene.addChild(x)
        else:
            tilesize = scene_info.tilesize
            shapegroup = {
                "type": "shapegroup",
                "id": "MainObject",
                "shape": object_value
            }

            shapegroup = self.pmgr.create(shapegroup)
            scene.addChild(shapegroup)
            for y_off in range(-tilesize, tilesize+1, 1):
                for x_off in range(-tilesize, tilesize+1, 1):
                    c_object_value = dict(object_value)

                    trans = mitsuba.Transform.translate(mitsuba.Vector(2*x_off, 2*y_off, 0))
                    # if "toWorld" in c_object_value:
                    #     c_object_value["toWorld"]=trans*c_object_value["toWorld"]
                    # else:
                    #     c_object_value["toWorld"]=trans
                    # x = self.pmgr.create(c_object_value)
                    # scene.addChild(x)
                    instance = {
                        "type": "instance",
                        "shapegroup": shapegroup,
                        "toWorld": trans
                    }

                    instance = self.pmgr.create(instance)
                    scene.addChild(instance)




        sceneResID = self.scheduler.registerResource(scene)

        return (scene, sceneResID), buffers




    def render(self, scene_info) -> np.ndarray:
        scene, sceneResID = scene_info
        queue = mitsuba_render.RenderQueue()
        job = mitsuba_render.RenderJob('MitsubaRenderJob', scene, queue, sceneResID)
        job.start()
        queue.waitLeft(0)
        queue.join()

        film = scene.getFilm()
        size = film.getSize()

        #bitmap = mitsuba.Bitmap(mitsuba.Bitmap.ERGB, mitsuba.Bitmap.EUInt8, size)
        bitmap = mitsuba.Bitmap(mitsuba.Bitmap.ERGB, mitsuba.Bitmap.EFloat32, size)

        film.develop(mitsuba.Point2i(0, 0), size, mitsuba.Point2i(0, 0), bitmap)


        buffer = bitmap.toByteArray()
        #buffer = np.frombuffer(buffer, dtype=np.uint8)
        buffer = np.frombuffer(buffer, dtype=np.float32)
        buffer = buffer.reshape(size[1], size[0], 3)
        #buffer = buffer/255

        self.scheduler.unregisterResource(sceneResID)
        #utils.tensor.displays(buffer)

        return buffer

