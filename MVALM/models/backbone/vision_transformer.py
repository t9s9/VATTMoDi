import torch
import torch.nn as nn

from .layers import LayerNorm
from .registry import register_model
from .transformer import Transformer


class VisionTransformer(nn.Module):
    """
    Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 gradient_checkpointing: bool = False, save_attn: bool = False, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.input_resolution = input_resolution
        self.first_run = True
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, attn_mask=None, activation=activation,
                                       gradient_checkpointing=gradient_checkpointing, save_attn=save_attn)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def initialize_parameters(self):
        self.transformer.initialize_parameters()

    def forward_feature(self, x: torch.Tensor):
        if self.first_run: print("ViT")
        if self.first_run: print("Input", x.shape)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        if self.first_run: print("After conv1", x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        if self.first_run: print("After reshape", x.shape)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        print("After embedding", x.shape)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.first_run: print("Before blocks", x.shape)
        x = self.transformer(x)
        if self.first_run: print("After blocks", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        if self.first_run: print("Out", x.shape)

        if self.first_run: self.first_run = False
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_feature(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


@register_model
def vit_base_patch16_224(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        'input_resolution': 224,
        'patch_size': 16,
        'width': vision_width,
        'layers': vision_layers,
        'heads': vision_heads,
        'output_dim': 512,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    model = VisionTransformer(**default_kwargs)
    return model


@register_model
def vit_base_patch32_224(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        'input_resolution': 224,
        'patch_size': 32,
        'width': vision_width,
        'layers': vision_layers,
        'heads': vision_heads,
        'output_dim': 512,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    model = VisionTransformer(**default_kwargs)
    return model


@register_model
def vit_large_patch14_224(**kwargs):
    vision_width = 1024
    vision_layers = 24
    vision_heads = vision_width // 64

    default_kwargs = {
        'input_resolution': 224,
        'patch_size': 14,
        'width': vision_width,
        'layers': vision_layers,
        'heads': vision_heads,
        'output_dim': 768,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    model = VisionTransformer(**default_kwargs)
    return model


@register_model
def vit_large_patch14_336(**kwargs):
    vision_width = 1024
    vision_layers = 24
    vision_heads = vision_width // 64

    default_kwargs = {
        'input_resolution': 336,
        'patch_size': 14,
        'width': vision_width,
        'layers': vision_layers,
        'heads': vision_heads,
        'output_dim': 768,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    model = VisionTransformer(**default_kwargs)
    return model
