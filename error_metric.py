import os
import utils
import torch
import skimage
import skimage.metrics
import sys
import path_config

try:
    import lpips

except ModuleNotFoundError as e:
    print("Skipping importing lpips")

try:
    import flip_loss
except ModuleNotFoundError as e:
    print("Skipping importing ")



def get_error(ground_path, result_path):
    ground = utils.exr.read(ground_path)
    result = utils.exr.read(result_path)

    ground = torch.Tensor(ground).unsqueeze(0).permute(0,3,1,2).float()
    result = torch.Tensor(result).unsqueeze(0).permute(0,3,1,2).float()

    # ground = ground[:,:, 53:512 - 53, :]
    # result = result[:,:, 53:512 - 53, :]

    em = ErrorMetric(ground, result, None, None, True)
    em.calc_errors()

    return em

class ErrorMetric:
    def __init__(self, ground, image, valid_pixels, path, compare):
        self.valid_pixels = valid_pixels

        self.ground = ground
        self.image = image

        if self.ground.shape[-1] > self.image.shape[-1]:

            ratio = self.ground.shape[-1]/self.image.shape[-1]

            self.ground = utils.tensor.blur(ratio/2, self.ground, True)

            self.ground = torch.nn.functional.interpolate(self.ground,
                                                   size=self.image.shape[-2:], mode='area')

        if self.valid_pixels is not None:
            assert False


        self.diff = self.image - self.ground

        self.compare = compare

        self.path = path

        #os.makedirs(path, exist_ok=True)

        if self.valid_pixels is None:
            self.valid_pixel_sum = self.ground.numel()
        else:
            self.valid_pixel_sum = int(self.valid_pixels.sum())

        print("self.valid_pixel_sum", self.valid_pixel_sum)

        def to_numpy(x: torch.Tensor):
            x = x.data.cpu().numpy()
            x = x[0,:,:,:].transpose(1,2,0)
            return x

        self.np_ground = to_numpy(self.ground)
        self.np_image = to_numpy(self.image)

        self.error_PSNR = "N/A"


    def string(self):


        self.error_LPIPS_vgg = self.calc_error_LPIPS("vgg")
        self.error_LPIPS_alex = self.calc_error_LPIPS("alex")

        self.error_FLIP = self.calc_error_FLIP()

        errors = {"L1": self.error_L1, "MSE": self.error_MSE,
                  "SSIM": self.error_SSIM, #"PSNR": self.error_PSNR,
                  "LPIPS_vgg": self.error_LPIPS_vgg,
                  "LPIPS_alex": self.error_LPIPS_alex,

                  "FLIP": self.error_FLIP,

                  }

        s = "\tL1:\t{L1:.6f}\tMSE:\t{MSE:.6f}\tSSIM:\t{SSIM:.6f}" \
            "\tLPIPS_vgg:\t{LPIPS_vgg:.6f}\tLPIPS_alex:\t{LPIPS_alex:.6f}" \
            "\tFLIP:\t{FLIP:.6f}".format(**errors)
        return s

    def save_index_page(self):
        path = os.path.join(self.path, "index.html")

        with open(path, "w") as f:
            errors = {"L1":self.error_L1, "MSE":self.error_MSE,
                      "SSIM":self.error_SSIM, "PSNR":self.error_PSNR}
            f.write("""
<pre>
L1: {L1:.6f}
MSE: {MSE:.6f}
SSIM: {SSIM:.6f}
PSNR: {PSNR}
</pre>
            """.format(**errors))

            f.write("""<img src='image.png'>Result<br>
            <img src='diff.png'>Diff<br>
            <img src='ground.png'>Ground<br>
            """)

            f.write("""
            <pre>
L1: {L1}
MSE: {MSE}
SSIM: {SSIM}
PSNR: {PSNR}
            </pre>
                        """.format(**errors))



    def save(self):
        pass
        #self.calc_errors()
        #utils.tensor.save_png(self.image, os.path.join(self.path, "image.png"))
        #utils.tensor.save_png(self.image, self.path + ".png")
        # utils.tensor.save_exr(self.image, self.path + ".exr")
        #
        # if self.compare:
        #     utils.tensor.save_png(self.ground, os.path.join(self.path, "ground.png"))
        #
        #     utils.tensor.save_png(torch.abs(self.diff), os.path.join(self.path, "diff.png"))
        #
        #     self.save_index_page()



    def calc_errors(self):
        self.error_L1 = self.calc_error_L1()
        self.error_MSE = self.calc_error_MSE()

        self.error_SSIM = self.calc_erro_SSIM()

        self.error_PSNR = None



        self.error_LPIPS_vgg = self.calc_error_LPIPS("vgg")
        self.error_LPIPS_alex = self.calc_error_LPIPS("alex")

        self.error_FLIP = self.calc_error_FLIP()

    def calc_error_L1(self):
        x = torch.abs(self.diff).sum()/self.valid_pixel_sum
        return float(x)


    def calc_error_FLIP(self):
        if self.image.shape[-1] <= 16: # Too small
            return float("NaN")

        def transform(x):
            x = x.clamp(0, 1)
            x = torch.pow(x, 1/2.2)
            x = x*2-1
            return x

        corrected_ground = transform(self.ground)
        corrected_image = transform(self.image)

        flip = flip_loss.FLIPLoss()

        loss = flip(corrected_image, corrected_ground)
        loss = float(loss)
        return loss



    def calc_error_LPIPS(self, network):
        if self.image.shape[-1] <= 16: # Too small
            return float("NaN")


        loss_fn_alex = lpips.LPIPS(net=network)

        def transform(x):
            x = x.clamp(0, 1)
            x = torch.pow(x, 1/2.2)
            x = x*2-1
            return x


        corrected_ground = transform(self.ground)
        corrected_image = transform(self.image)

        loss = loss_fn_alex(corrected_image, corrected_ground)
        loss = float(loss)
        return loss





    def calc_error_MSE(self):
        diff2 = self.diff*self.diff
        x = diff2.sum()/self.valid_pixel_sum
        return float(x)

    def calc_erro_SSIM(self):
        if self.image.shape[-1] <= 8: # Too small
            return float("NaN")

        x = skimage.metrics.structural_similarity(self.np_ground , self.np_image, multichannel=True)
        return float(x)




    def save_diff(self, path):
        direction  = self.diff.mean(-3, keepdims=True)
        pos_dir = direction.clamp(0)*4
        neg_dir = (-direction).clamp(0)*4

        result = torch.zeros_like(self.diff)

        result[:, 0:1, :,:] = 1 - pos_dir
        result[:, 1:2, :,:] = 1 - neg_dir - pos_dir
        result[:, 2:3, :,:] = 1 - neg_dir
        utils.tensor.save_png(result, path)

def main():
    assert len(sys.argv) == 3

    ground_path = path_config.path_images + "/" + sys.argv[1]
    result_path = path_config.path_images + "/" + sys.argv[2]

    er = get_error(ground_path, result_path)

    print(er.string())


if __name__ == "__main__":
    main()

