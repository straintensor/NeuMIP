
import cpputils
import utils
import torch


def util_grid_sample(grid, input):
    with utils.Timer("Permute"):
        grid = grid.permute(0,2,3,1)
    input = input.unsqueeze(0)

    return utils.tensor.grid_sample(input, grid, padding_mode="repeat")


def main():
    device = torch.device("cuda:0")



    texture_tensor_0 = torch.rand((3, 64,64), dtype=torch.float32, device=device)

    texture_tensor_0 = utils.tensor.load_image("/home/alex/projects/neural-mipmap/assets/images/lion.png")

    texture_tensor_0 = torch.Tensor(texture_tensor_0)
    texture_tensor_0 = texture_tensor_0.permute(2,0,1)
    texture_tensor_0 = texture_tensor_0.float().to(device=device)

    num_ch, height, width = texture_tensor_0.shape


    grid_x = torch.linspace(-1+1/width,1-1/width,width)
    grid_y = torch.linspace(-1+1/height,1-1/height,height)+20/height

    grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
    uv_tensor = torch.stack([grid_x, grid_y], 0)
    uv_tensor = uv_tensor.unsqueeze(0)
    uv_tensor = uv_tensor.float().to(device=device)


    with utils.Timer("PyTorch"):
        result_classic = util_grid_sample(uv_tensor, texture_tensor_0)


    with utils.Timer("CUDA"):
        result = cpputils.texture_interpolate(uv_tensor, texture_tensor_0)
    #
    # utils.tensor.displays(texture_tensor_0)
    # utils.tensor.displays(torch.abs(texture_tensor_0-result))

    utils.tensor.displays(result)
    utils.tensor.displays(result_classic)
    utils.tensor.displays(torch.abs(result_classic-result)*10)



if __name__ == "__main__":
    main()