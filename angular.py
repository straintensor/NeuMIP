import utils
import torch
import numpy as np

class Angular:


    def __init__(self, camera_dir, light_dir):
        self.camera_dir = camera_dir
        self.light_dir = light_dir
        self.diff_dir = None
        self.half_dir = None

    num_channels = 0

    def base_convert_to_3D(self):
        self.light_dir = self.convert_to_3D(self.light_dir)
        self.camera_dir = self.convert_to_3D(self.camera_dir)

    def aux_convert_to_3D(self):
        self.diff_dir = self.convert_to_3D(self.diff_dir)
        self.half_dir = self.convert_to_3D(self.half_dir)

    def calc_aux(self):

        #self.diff_dir =
        self.half_dir = utils.la4.normalize(self.light_dir + self.camera_dir)


    def convert_to_2D(self, dir):
        if dir is None:
            return None

        return dir[:, :2, :, :]


    def base_convert_to_2D(self):
        self.light_dir = self.convert_to_2D(self.light_dir)
        self.camera_dir = self.convert_to_2D(self.camera_dir)

    def aux_convert_to_2D(self):
        self.diff_dir = self.convert_to_2D(self.diff_dir)
        self.half_dir = self.convert_to_2D(self.half_dir)

    def convert_to_3D(self, dir):
        if dir is None:
            return None

        if dir.shape[1] == 2:
            dir = torch.cat([dir, utils.la4.get_3rd_axis(dir)], dim=1)
        return dir

    def convert_to_fourier(self, dir, levels):
        dirs = []


        for level in range(levels):
            if level ==0:
                dirs.append(dir)
            else:
                dirs.append(torch.cos(dir*np.pi * 2**level))
                dirs.append(torch.sin(dir*np.pi * 2**level))

        dir = torch.cat(dirs, dim=1)

        return dir

    def check_ch(self, x):
        assert type(self).num_channels == x.shape[1]
        return x

class AngularSimple(Angular):

    num_channels = 4


    def convert(self):
        # print(self.light_dir.shape)
        # print(self.camera_dir.shape)
        x = torch.cat([self.light_dir, self.camera_dir], dim=1)

        return self.check_ch(x)
