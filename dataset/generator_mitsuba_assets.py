try:
    import mitsuba.core as mitsuba

except ModuleNotFoundError as e:
    print("Skipping importing mitsuba")

import path_config
import os

def get_textured_material(base_path):
    tfiles = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            tfiles.append(os.path.join(base_path, file))
    del files

    def get_path(prefix):
        for file in tfiles:
            if prefix in file:
                return file



    def get_bitmap(path):
        return {
            "type": "bitmap",
            "name": "reflectance",
            "warpMode": "repeat",
            "filename": path,
            "vscale": -1.,
            "uscale": 1.,

        }

    tex_diff_path = get_path("_diff_1k")
    tex_rough_path = get_path("_rough_1k")

    material_value = {
        "specularReflectance": get_bitmap(tex_diff_path),
        # "type": "diffuse",
        "type": "roughconductor",
        "distribution": "ggx",
        "material": "Ag",
        "alpha": get_bitmap(tex_rough_path)

    }

    return material_value


def get_material(material):
    path_assets = path_config.path_assets + "/"

    if material == "ggx.5":
        material_value = {
            "specularReflectance": mitsuba.Spectrum([1. ,1., 0.]),
            # "type": "diffuse",
            "type": "roughconductor",
            "distribution": "ggx",
            "material": "Ag",
            "alpha": 0.1

        }

    elif material == "grid128g3":
        material_value = {
            "specularReflectance": {
                "type": "bitmap",
                "name": "reflectance",
                "warpMode": "repeat",
                "filename": path_assets + "textures/grid128.png"

            },
            # "type": "diffuse",
            "type": "roughconductor",
            "distribution": "ggx",
            "material": "Ag",
            "alpha": 0.3

        }
    elif material == "grid128g1":
        material_value = {
            "specularReflectance": {
                "type": "bitmap",
                "name": "reflectance",
                "warpMode": "repeat",
                "filename": path_assets + "textures/grid128.png"

            },
            # "type": "diffuse",
            "type": "roughconductor",
            "distribution": "ggx",
            "material": "Ag",
            "alpha": 0.1

        }

    elif material == "grid128g4v":
        material_value = {
            "specularReflectance": {
                "type": "bitmap",
                "name": "reflectance",
                "warpMode": "repeat",
                "filename": path_assets + "textures/grid128.png"

            },
            # "type": "diffuse",
            "type": "roughconductor",
            "distribution": "ggx",
            "material": "Ag",
            "alphaV": 0.4,
            "alphaU": 0.1

        }


    elif material == "neuralmat":
        material_value = {
            "reflectance": mitsuba.Spectrum(0.0),
            "type": "neuralmat",

        }
    elif material.startswith("ubo2014:"):
        btf_name = material.split(":", 1)[1]
        material_value = {
            "type": "ubo2014",
            "ubo_path": path_config.path_ubo2014 + "/" + btf_name

        }

    elif material == "diffuse":
        material_value = {
            "reflectance": mitsuba.Spectrum(0.0),
            "type": "diffuse",

        }
    elif material == "white":
        material_value = {
            "reflectance": mitsuba.Spectrum(1.0),
            "type": "diffuse",

        }
    elif material == "grid128":
        material_value = {
            "reflectance": {
                "type": "bitmap",
                "name": "reflectance",
                "warpMode": "repeat",
                "filename": path_assets + "textures/grid128.png"

            },
            "type": "diffuse",

        }
    elif material == "grid128d":
        material_value = {
            "reflectance": {
                "type": "bitmap",
                "name": "reflectance",
                "warpMode": "repeat",
                "filename": path_assets + "textures/grid128.png"

            },
            "type": "diffuse",
            "debug_inc": True,
        }


    else:
        material_value = get_textured_material(path_assets + "shapes/" + material + "/")

    return material_value



def get_object(object_name, material_value):

    path_assets = path_config.path_assets + "/"

    ply_val = {"type": "ply",
                        "collapse": True,
                        "loadMaterials": False,
                        "bsdf": material_value,
                        }

    if object_name == "plane1":
        object_type = 'rectangle'

        object_value = {
            'type': 'rectangle',


            'bsdf': material_value
        }

    else:
        object_type = 'shape'
        object_value = ply_val
        object_value["filename"] = path_assets + "shapes/" + object_name + "/model.ply"

    return object_type, object_value



