from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image

import utils.exr
import time

def to_output_format(x, norm=False, raw=False):
    if isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()



    if len(x.shape) == 4:
        x = x[0, ...]

    if len(x.shape) == 2:
        x = x[..., np.newaxis]
        x

    assert len(x.shape) == 3

    if x.shape[0] < x.shape[2]:
        x = x.transpose(1, 2, 0)

    if x.shape[2] < 3:
        prev_size = x.shape[2]
        #x = np.repeat(x, 3, 2)
        x = np.tile(x, [1,1,3])
        if prev_size == 2:
            x[:,:,2] = 0


    x = x[:,:,:3]

    if norm:
        x = x*.5+.5

    #x[:, :, 2] = 0
    if not raw:
        x = np.clip(x, 0, 1)
        x = np.power(x, 1/2.2)
    return x

def displays(x: np.ndarray, norm=False):
    x = to_output_format(x, norm)
    x[0,0,0]= 1/10

    sizes = x.shape[:2]

    fig = plt.figure()

    dpi = 72
    fig.set_dpi(dpi)
    fig.set_size_inches( sizes[1]/dpi ,  sizes[0]/dpi, forward=False)


    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(x, vmin=0, vmax=1)
    plt.margins(y=0)
    plt.axis('off')
    plt.show()
    plt.close(fig)



def save_png(x, path):
    x =  to_output_format(x, False, False)
    x = x*255
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(path)



def save_exr(x, path):
    x =  to_output_format(x, False, True)
    utils.exr.write16(x, path)


class Type:
    def __init__(self, x):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype

    def same_type(self):
        return {"device": self.device,
                "dtype": self.dtype
                }



def zeros_like(x, shape):
    return torch.zeros(shape, **Type(x).same_type())

def ones_like(x, shape):
    return torch.ones(shape, **Type(x).same_type())


def blur2(sigma, x):
    num_ch = x.shape[-3]

    radius = int(np.ceil(1.5*sigma))

    if sigma < .8:
        return x

    axis = np.linspace(-radius, radius, 2*radius+1)
    yy, xx = np.meshgrid(axis, axis)

    kernel = np.exp(- (xx*xx + yy*yy) / (sigma*sigma*2))

    kernel = kernel/kernel.sum()
    kernel = torch.tensor(kernel, **Type(x).same_type()).unsqueeze(0).unsqueeze(0)

    if False:

        x = [torch.nn.functional.conv2d(x[:, idx:idx+1, :, :], kernel, padding=radius) for idx in range(num_ch)]
        x = torch.cat(x, -3)

    else:
        big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], **Type(x).same_type())
        for idx in range(num_ch):
            big_kernel[idx:idx+1, idx:idx+1, :, :] = kernel

        x = torch.nn.functional.conv2d(x, big_kernel, padding=radius)

    return x

def blur1d(sigma, x, hor=False, warp=True, sm=1.5):
    num_ch = x.shape[-3]

    radius = int(np.ceil(sm*sigma))

    if sigma < .2:
        return x

    xx = np.linspace(-radius, radius, 2*radius+1)

    kernel = np.exp(- (xx*xx) / (sigma*sigma*2))

    if hor:
        exp_dim = -2
        padding = (0, radius)
    else:
        exp_dim = -1
        padding = (radius, 0)

    if warp:
        padding = 0

    kernel = kernel/kernel.sum()
    kernel = torch.tensor(kernel, **Type(x).same_type()).unsqueeze(0).unsqueeze(0).unsqueeze(exp_dim)

    if False:

        x = [torch.nn.functional.conv2d(x[:, idx:idx+1, :, :], kernel, padding=padding) for idx in range(num_ch)]
        x = torch.cat(x, -3)

    else:
        big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], **Type(x).same_type())
        for idx in range(num_ch):
            big_kernel[idx:idx+1, idx:idx+1, :, :] = kernel

        if warp:
            if hor:
                size = x.shape[-1]
            else:
                size = x.shape[-2]

            repeat = int(radius/size)
            rr = radius%size

            repeat = repeat*2 + 1

            items = [x] * repeat


            if hor:
                items.append(x[:,:,:,:rr])
                if rr >0:
                    items = [x[:,:,:,-rr:]] + items
                x = torch.cat(items, -1)
            else:
                items.append(x[:,:,:rr,:])
                if rr >0:
                    items = [x[:,:,-rr:,:]] + items

                x = torch.cat(items, -2)


        x = torch.nn.functional.conv2d(x, big_kernel, padding=padding)

    return x

def blur(sigma,x, warp, sm=3):
    #x2 = blur2(sigma, x)
    #return x
    x = blur1d(sigma, x, False, warp, sm)
    x = blur1d(sigma, x, True, warp, sm)



    return x


def get_masks(probabilities, shape):
    probabilities = np.array(probabilities)
    cum_prob = np.cumsum(probabilities)
    cum_prob = cum_prob/cum_prob[-1]
    cum_prob = np.insert(cum_prob, 0, 0., axis=0)

    rand = torch.rand(shape) #, **type.same_type())
    masks = []

    for i in range(len(cum_prob)-1):
        mask = torch.logical_and(cum_prob[i] < rand, rand <= cum_prob[i+1])
        masks.append(mask)

    return masks




def blur2(radius, x):
    texture = \
    torch.nn.functional.interpolate(x, scale_factor=(scale_factor, scale_factor), mode='area')





def fmod(x):
    x = torch.remainder(x, 1.)
    return x

def fmod1_1(x):
    x = x * .5 + .5
    x = fmod(x)
    x = x * 2 - 1
    return x

# grid: [b, h, w, 2]
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', compensate=True):
    # padding_mode = 'repeat'
    #assert (padding_mode) == "repeat"
    #assert (compensate) == True

    prev_padding_mode = padding_mode

    if padding_mode == 'repeat':
        grid = fmod1_1(grid)
        padding_mode = 'border'
    else:
        pass
       # assert False

    # speed
    # align_corners = True
    # result = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
    #                                          align_corners=align_corners)
    # return result
    if compensate:

        grid_x = grid[:,:,:,0:1]
        grid_y = grid[:,:,:,1:2]

        align_corners = True

        if prev_padding_mode == "repeat":
            input = pad_around_1(input)
            # print(input.shape)
            height, width = input.shape[2:] # modified
            grid_x = grid_x*(width-2)/(width-1)
            grid_y = grid_y*(height-2)/(height-1)

        else:
            height, width = input.shape[2:]
            grid_x = grid_x*(width)/(width-1)
            grid_y = grid_y*(height)/(height-1)

        grid = torch.cat([grid_x, grid_y], dim=-1)
    else:
        assert False
        align_corners = False

    # mode = "nearest"
    # padding_mode = 'zeros'
    result = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return result

def generate_grid(height, width):

    half_hor = 1./width
    half_ver = 1./height

    x = np.linspace(-1 + half_hor, 1 - half_hor, width)
    y = np.linspace(-1 + half_ver, 1 - half_ver, height)

    xv, yv = np.meshgrid(y, x)
    location = np.stack([xv, yv], 0)
    location = torch.Tensor(location).float()
    location = location.unsqueeze(0)
    return location


def pad_around_1(x):
    # shape = x.shape
    #
    # x = torch.nn.functional.pad(x, (1,1,1,1), 'circular')
    # return x

    x = torch.cat([x[:,:,:,-1:], x, x[:,:,:,:1]], dim=-1)
    x = torch.cat([x[:,:,-1:,:], x, x[:,:,:1,:]], dim=-2)


    return x


def to_device(device, *xs):
    res =  [x.to(device=device) for x in xs]
    if len(res) == 1:
        return res[0]

    return res


def mse_loss(weight, result, ground):
    diff = (result - ground)
    diff = diff * diff

    diff = diff*weight
    return diff.mean()

def l1_loss(weight, result, ground):
    diff = torch.abs(result - ground)

    diff = diff*weight
    return diff.mean()



def yuv_to_rgb(x):
    assert len(x.shape) == 4

    def to_tensor(c):
        c = torch.Tensor(c)
        c = c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        c = c.float()
        c = c.to(device=x.device)
        return c

    to_R = to_tensor([1, 0, 1.13983])
    to_G = to_tensor([1, -0.39465,   -0.58060])
    to_B = to_tensor([1, 2.03211, 0])


    def multiplier(c):
        r = x*c
        r = r.sum(dim=1, keepdims=True)
        return r

    x = torch.cat([multiplier(to_R),multiplier(to_G), multiplier(to_B)], dim=1)
    return x

def load_image( filename ) :
    img = Image.open( filename )
    img.load()
    data = np.asarray( img, dtype="float32" )/255

    return data



def assert_shape(x, shape):
    assert len(x.shape) == len(shape)

    for idx, size in enumerate(shape):
        if size is not None and size != -1:
            assert x.shape[idx] == size


def convert_la4_to_linear(x: torch.Tensor):
    assert len(x.shape) == 4
    x = x.permute(0,2,3,1)
    num_ch = x.shape[-1]
    x = x.reshape(-1, num_ch)
    return x


def get_shape(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return list(x.shape)
    else:
        return list(x)

def convert_linear_to_la4(x: torch.Tensor, example: torch.Tensor):
    shape = get_shape(example)

    shape[1] = x.shape[-1]
    new_shape = [shape[0], shape[2], shape[3], shape[1]]

    x = x.reshape(*new_shape)
    x = x.permute(0,3,1,2)
    return x