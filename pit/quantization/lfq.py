# From https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/vqvae/lookup_free_quantize.py

from math import log2, ceil
from collections import namedtuple

import torch
from torch import einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce

# constants

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)

# class

def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y

def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))

def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])

        avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss

class LFQQuantizer(Module):
    def __init__(
        self,
        format,
        dim = None,
        codebook_size = None,
        num_codebooks = 1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    ):
        super().__init__()

        self.format = format
        assert self.format in ["bchw", "blc"]

        # some assert validations
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        self.codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_dim = int(log2(codebook_size))

        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = self.codebook_dim
        self.num_codebooks = num_codebooks
        
        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        self.register_buffer('mask', 2 ** torch.arange(self.codebook_dim), persistent=False)
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes
        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0
        self.register_buffer('codebook', codebook, persistent = False)
    
    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x
    
    def dequant(self, x):
        return x

    def forward(
        self,
        x,
        mask = None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        if self.format == "bchw":
            b, c, h, w = x.shape
            ndim = c * h * w
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            b, l, c = x.shape
            ndim = l * c

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value) # higher than 0 filled 

        # entropy aux loss

        if self.training:
            logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits = logits,
                sample_minimization_weight = self.sample_minimization_weight,
                batch_maximization_weight = self.batch_maximization_weight
            )
            avg_probs = self.zero
        else:
            # if not training, just return dummy 0
            per_sample_entropy = codebook_entropy = self.zero
            ## calculate the codebook_entropy needed for one batch evaluation
            entropy_aux_loss = self.zero
            avg_probs = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(x, quantized.detach(), reduction = 'none')
            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero


        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = x + (quantized - x).detach() #transfer to quantized

        # merge back codebook dim

        quantized = rearrange(quantized, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions

        if self.format == "bchw":
            quantized = rearrange(quantized, "b (h w) c -> b c h w", h=h)

        info = {
            "entropy_aux_loss": entropy_aux_loss,
            "per_sample_entropy": per_sample_entropy.detach(),
            "codebook_entropy": codebook_entropy.detach(),
            "commit_loss": commit_loss,
        }
        return quantized, info

if __name__ == "__main__":
    quantizer = LFQQuantizer(
    format = "bchw",
    codebook_size = 2**8,      # codebook size, must be a power of 2
    dim = 8,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
    num_codebooks = 2,
    sample_minimization_weight = 1.0,        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    batch_maximization_weight = 1.0
)

    image_feats = torch.randn(2, 16, 16, 16) #16 is dim, must be power of 2 of codebook_size

    quantized, info = quantizer(image_feats)  # you may want to experiment with temperature
    print("quantized shape:", quantized.shape)