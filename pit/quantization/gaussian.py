import torch
import torch.nn as nn
from einops import rearrange


class GaussianRegularizer(nn.Module):
    # Gaussian VAE

    # args
    # levels: the FSQ levels parameter, see Table 1 of FSQ paper
    # format: data format, must be one of []"bchw", "blc"]

    def __init__(self, format, logvar_range=[-30.0, 20.0]):
        super().__init__()

        self.format = format

        assert self.format in ["bchw", "blc"]
        self.logvar_range = logvar_range

    def forward(self, z):
        z = z.float()

        if self.format == "bchw":
            b, c, h, w = z.shape
            ndim = c * h * w
            zhat = rearrange(z, "b c h w -> b (h w) c")
        else:
            b, l, c = z.shape
            ndim = l * c
            zhat = z

        # b, l, c
        mu, logvar = zhat.chunk(2, 2)
        logvar = torch.clamp(logvar, self.logvar_range[0], self.logvar_range[1])

        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)

        zhat = mu + torch.randn_like(mu) * std

        kl = 0.5 * torch.sum(
            torch.pow(mu, 2) + var - 1.0 - logvar,
            dim=[1, 2],
        )

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)

        info = {"kl": torch.mean(kl)}
        return zhat, info


if __name__ == "__main__":
    z = torch.randn([1, 2, 4, 4]).cuda()
    gauss = GaussianRegularizer("bchw").cuda()
    zhat, info = gauss(z)

    print(info)
