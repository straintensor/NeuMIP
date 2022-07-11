import torch
import numpy as np

class InputInfo:
    def __init__(self):
        self.light_dir_x = None
        self.light_dir_y = None

        self.camera_dir_x = None
        self.camera_dir_y = None

        self.mipmap_level_id = None

    def to_torch(self, resolution, use_repeat=True):
        #assert False
        height, width = resolution

        def convert_item(x):
            if not isinstance(x, torch.Tensor):
                if isinstance(x, np.ndarray):
                    x = torch.Tensor(x)
                else:
                    x = torch.Tensor([x])

            if len(x.shape) == 1:
                x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            if use_repeat:
                if x.shape[-1] == 1 and x.shape[-2] == 1:


                    x = x.repeat([1, 1, height, width])
            x = x.float().cpu()
            return x

        self.light_dir_x = convert_item(self.light_dir_x)
        self.light_dir_y = convert_item(self.light_dir_y)


        self.camera_dir_x = convert_item(self.camera_dir_x)
        self.camera_dir_y = convert_item(self.camera_dir_y)

        self.mipmap_level_id = convert_item(self.mipmap_level_id)
