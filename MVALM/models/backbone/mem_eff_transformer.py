import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

from .attention import MemoryEfficientAttention
from .cross_attn import CrossAttention

from xformers.ops.fmha import MemoryEfficientAttentionFlashAttentionOp


class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 dropout: float = 0.0,
                 activation: nn.Module = nn.GELU,
                 hidden_layer_multiplier: int = 4,
                 bias: bool = True):
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim_mlp, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(in_features=dim_mlp, out_features=dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 mlp_hidden_layer_multiplier: int = 4,
                 mlp_dropout: float = 0.0,
                 mlp_activation: nn.Module = nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attention = MemoryEfficientAttention(dim=dim,
                                                  num_heads=num_heads,
                                                  qkv_bias=qkv_bias,
                                                  attn_drop=attn_drop,
                                                  proj_drop=proj_drop)

        # self.attention = CrossAttention(query_dim=dim,
        #                                 cross_attention_dim=dim,
        #                                 heads=num_heads,
        #                                 dim_head=dim // num_heads,
        #                                 bias=qkv_bias,
        #                                 dropout=proj_drop)
        # self.attention.set_use_memory_efficient_attention_xformers(True, MemoryEfficientAttentionFlashAttentionOp)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_layer_multiplier=mlp_hidden_layer_multiplier, dropout=mlp_dropout,
                       activation=mlp_activation)

    def forward(self, x: torch.Tensor):
        # x = x + self.attention(self.ln1(x), attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_layers: int,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 mlp_hidden_layer_multiplier: int = 4,
                 mlp_dropout: float = 0.0,
                 mlp_activation: nn.Module = nn.GELU,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        print("Using gradient checkpointing: ", gradient_checkpointing)

        self.blocks = nn.Sequential(
            *[AttentionBlock(dim, num_heads, qkv_bias, attn_drop, proj_drop, mlp_hidden_layer_multiplier, mlp_dropout,
                             mlp_activation) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint_sequential(functions=self.blocks, input=x, segments=self.num_layers)
        return self.blocks(x)
