import torch

from .metric import PSNR, SSIM, LPIPS


class Measure:
    def __init__(self, metrics, net_lpips="alex", device="cpu", dtype=torch.float32):
        self.metrics = metrics
        if "psnr" in self.metrics:
            self.psnr = PSNR()
        if "ssim" in self.metrics:
            self.ssim = SSIM(device=device, dtype=dtype)
        if "lpips" in self.metrics:
            self.lpips = LPIPS(net=net_lpips, device=device, dtype=dtype)

    def __call__(self, im1, im2, data_range):
        result = {}
        if "psnr" in self.metrics:
            result["psnr"] = self.psnr(im1, im2, data_range=data_range)
        if "ssim" in self.metrics:
            result["ssim"] = self.ssim(im1, im2, data_range=data_range)
        if "lpips" in self.metrics:
            result["lpips"] = self.lpips(im1, im2)
        return result
