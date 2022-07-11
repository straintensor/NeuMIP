import os
import socket

class Config:
    def __init__(self, hostname):
        self.hostname = hostname

        if hostname=="default_config":
            path_base = "/home/alex/projects/neural-mipmap"
        else:
            assert False

        self.path_base = path_base

        self.path_dataset = os.path.join(path_base, "datasets")
        self.path_dataset_pbrt = os.path.join(path_base, "datasets_pbrt")
        self.path_models = os.path.join(path_base, "models")
        self.path_assets = os.path.join(path_base, "assets")
        self.path_videos = os.path.join(path_base, "videos")
        self.path_images = os.path.join(path_base, "images")
        self.path_export = os.path.join(path_base, "export")
        self.path_ubo2014 = os.path.join(path_base, "ubo2014")

        self.copy_dataset = False

default_config = Config("default_config")



def set_local(config):
    for var_name in vars(config):
        if var_name.startswith("path_") or var_name.startswith("copy_") :
            globals()[var_name] = getattr(config, var_name)


def get_value(type_name, path):


    return "/".join((globals()[type_name], path))

hostname = socket.gethostname()

set_local(default_config)

