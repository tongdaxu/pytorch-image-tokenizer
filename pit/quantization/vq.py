import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class VQQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        format,
        n,
        dim,
        beta=0.25,
        codebook_num=1,
        legacy=True,
    ):
        super().__init__()
        self.format = format
        self.n = n
        self.dim = dim
        self.beta = beta
        self.legacy = legacy
        self.codebook_num = codebook_num

        self.embedding = nn.Embedding(self.n, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n, 1.0 / self.n)

    def get_trainable_parameters(self):
        return self.embedding.parameters()

    def forward(self, z):
        if self.format == "bchw":
            b, c, h, w = z.shape
            ndim = c * h * w
            z = rearrange(z, "b c h w -> b h w c").contiguous()
        else:
            b, l, c = z.shape
            ndim = l * c
            h = int(np.sqrt(l))
            assert h * h == l, "Input length must be a perfect square for blc format"
            z = rearrange(z, "b l c -> b c h h", h=h).contiguous()

        z_flattened = z.view(-1, self.dim, self.codebook_num)
        z_q = []
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        for i in range(self.codebook_num):
            d = (
                torch.sum(z_flattened[:, :, i] ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2
                * torch.einsum(
                    "bd,dn->bn",
                    z_flattened[:, :, i],
                    rearrange(self.embedding.weight, "n d -> d n"),
                )
            )

            min_encoding_indices = torch.argmin(d, dim=1)
            z_q_i = self.embedding(min_encoding_indices)
            z_q.append(z_q_i[:, :, None])
        z_q = torch.cat(z_q, dim=2)
        z_q = z_q.view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if self.format == "bchw":
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        else:
            z_q = rearrange(z_q, "b h w c -> b (h w) c").contiguous()

        return z_q, {"codebook_loss": loss}

    def get_codebook_entry(self, indices, shape):
        raise NotImplementedError


if __name__ == "__main__":
    # Example usage
    vq = VQQuantizer(format="bchw", n=65535, dim=4)
    z = torch.randn(2, 16, 32, 32)  # Example input tensor
    z_q, info = vq(z)
    print("Quantized shape:", z_q.shape)
    print("Info:", info)
