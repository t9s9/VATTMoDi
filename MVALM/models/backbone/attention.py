import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

is_xformers_available = importlib.util.find_spec("xformers") is not None
if is_xformers_available:
    import xformers
    import xformers.ops
else:
    xformers = None


class MemoryEfficientAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ):
        super().__init__()
        assert (dim % num_heads == 0), 'dim should be divisible by num_heads'
        assert num_heads > 0

        self.dim = dim
        self.num_heads = num_heads
        self.dim_heads = dim // num_heads
        self.attn_drop = attn_drop
        self.scale = self.dim_heads ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=False)

        self.store_attn = False
        self.attn = None

        self._use_memory_efficient_attention_xformers = False
        self.set_use_memory_efficient_attention_xformers(True)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available:
                print("Warning: xformers is not available, falling back to slow attention implementation. "
                      "Refer to https://github.com/facebookresearch/xformers for more information on how to install.")
                return False
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e
        print("Using memory efficient attention" if use_memory_efficient_attention_xformers else "Using slow attention")
        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, x, attention_mask=None):
        # B : batch size
        # S : sequence length
        # D : embedding size
        # H : number of heads
        # K : embeddings size per head
        B, N, C = x.size()  # B x S x D

        qkv = (
            self.qkv(x)  # B x S x (3*D)
            .reshape(B, N, 3, self.num_heads, self.dim_heads)  # B x S x 3 x H x D/H
            .permute(2, 0, 3, 1, 4)  # 3 x B x H x S x D/H
            .flatten(1, 2)  # 3 x (B*H) x S x D/H
        )

        q, k, v = qkv.unbind()  # (B*H) x S x D/H

        # print("Attention shape:", q.shape, k.shape, v.shape) torch.Size([3072, 402, 64])
        if self._use_memory_efficient_attention_xformers:
            x = xformers.ops.memory_efficient_attention(
                q, k, v, p=self.attn_drop, attn_bias=attention_mask,
                op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            )
        else:
            # todo implement using torch.baddbmm
            attn = (self.scale * q) @ k.transpose(-2, -1)
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = attn.softmax(-1)
            if self.store_attn:
                self.attn = attn
            attn = F.dropout(attn, self.attn_drop)
            x = attn @ v

        x = (
            x  # (B*H) x S x D/H
            .view(B, self.num_heads, N, self.dim_heads)  # B x H x S x D/H
            .transpose(1, 2)  # B x S x H x D/H
            .reshape(B, N, C)  # B x S x D
        )

        x = self.proj_drop(self.proj(x))
        return x
