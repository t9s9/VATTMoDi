from collections import OrderedDict
from typing import Tuple, Optional

import torch
import torch.nn as nn

# from fairscale.nn import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint_sequential

from .layers import LayerNorm, QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, save_attn: bool = False,
                 activation: nn.Module = QuickGELU()):
        super().__init__()
        self.n_head = n_head
        self.save_attn = save_attn
        self.attn_gradients = None
        self.attention_map = None

        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", activation),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_output, attn_output_weights = self.attn(x, x, x, need_weights=self.save_attn, attn_mask=self.attn_mask)
        if self.save_attn:
            self.save_attention_map(attn_output_weights)
            attn_output_weights.register_hook(self.save_attn_gradients)

        return attn_output

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))  # , register_hook=self.save_attn)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 gradient_checkpointing: bool = False, save_attn: bool = False, activation: nn.Module = QuickGELU()):
        super().__init__()
        self.width = width
        self.layers = layers
        self.gradient_checkpointing = gradient_checkpointing
        print("Using gradient checkpointing: ", gradient_checkpointing)

        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, save_attn, activation=activation) for _ in range(layers)]
        )

        # if self.gradient_checkpointing:
        #     print("Using gradient checkpointing.")
        #     self.resblocks = checkpoint_wrapper(self.resblocks, offload_to_cpu=False)

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing and self.training:
            return checkpoint_sequential(functions=self.resblocks, input=x, segments=self.layers)
        return self.resblocks(x)

    def initialize_parameters(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
