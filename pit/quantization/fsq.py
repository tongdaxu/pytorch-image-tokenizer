import torch
import torch.nn as nn
from einops import rearrange


def round_ste(z):
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQQuantizer(nn.Module):
    # FSQQuantizer
    # follow the Appendix A.1 of FSQ paper
    # in https://arxiv.org/pdf/2309.15505
    # only operates with z \in [-1, +1]

    # args
    # levels: the FSQ levels parameter, see Table 1 of FSQ paper
    # format: data format, must be one of []"bchw", "blc"]

    def __init__(self, levels, format):
        super().__init__()
        self.levels = nn.Parameter(
            torch.tensor(levels, dtype=torch.int32), requires_grad=False
        )
        self.dim = self.levels.shape[0]
        self.format = format

        assert self.format in ["bchw", "blc"]

    def _quantize(self, zhat, eps=1e-3):
        half_l = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        print(zhat.shape, shift.shape, half_l.shape, offset.shape)
        bounded_z = (zhat + shift).tanh() * half_l - offset

        half_width = self.levels // 2

        round_bounded_z = round_ste(bounded_z)
        zhat = round_bounded_z / half_width
        indices = round_bounded_z + half_width

        return zhat, indices.to(dtype=torch.int32)

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

        zhat, indices = self._quantize(zhat)

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)
            indices = rearrange(indices, "b (h w) c -> b c h w", h=h)

        info = {"indices": indices, "bits": torch.sum(torch.log2(self.levels)) * ndim}
        return zhat, info

    def dequant(self, indices):
        if self.format == "bchw":
            b, c, h, w = indices.shape
            indices = rearrange(indices, "b c h w -> b (h w) c")

        half_width = self.levels // 2
        zhat = (indices - half_width) / half_width

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)

        return zhat

    def generate(self, shape):
        if self.format == "bchw":
            shape_bl = [shape[0], shape[2] * shape[3], 1]
        else:
            shape_bl = [shape[0], shape[1], 1]
        indices = []
        for level in self.levels:
            indice = torch.randint(0, level, shape_bl).to(device=self.levels.device)
            indices.append(indice)
        indices = torch.cat(indices, dim=2)
        if self.format == "bchw":
            indices = rearrange(indices, "b (h w) c -> b c h w", h=shape[2])
        return self.dequant(indices)


if __name__ == "__main__":
    z = torch.randn([16, 2, 4, 4]).cuda()
    fsq = FSQQuantizer([8, 8], "bchw").cuda()
    zhat, info = fsq(z)
    zhat2 = fsq.dequant(info["indices"])

    zhat_gen = fsq.generate([16, 2, 4, 4])

    print("quant error", torch.mean(torch.abs(z - zhat2)))
    print("dequant error", torch.mean(torch.abs(zhat - zhat2)))
    print(info["bits"])
