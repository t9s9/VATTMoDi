from collections import OrderedDict

import torch
import torch.nn as nn
# from fairscale.nn import checkpoint_wrapper

from .registry import register_model
from .layers import AttentionPool2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, checkpoint: bool = False):
        super().__init__()
        print("Initialise ResNet with checkpoint =", checkpoint)
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.checkpoint = checkpoint

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def initialize_parameters(self):
        std = self.attnpool.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        if self.checkpoint:
            return checkpoint_wrapper(nn.Sequential(*layers), offload_to_cpu=False)
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


@register_model
def modified_resnet_50(**kwargs):
    vision_width = 64
    vision_heads = vision_width * 32 // 64
    default = {
        'layers': [3, 4, 6, 3],
        'output_dim': 1024,
        'heads': vision_heads,
        'width': vision_width,
        'input_resolution': 224
    }
    default.update(**kwargs)
    return ModifiedResNet(**default)


@register_model
def modified_resnet_101(**kwargs):
    vision_width = 64
    vision_heads = vision_width * 32 // 64
    default = {
        'layers': [3, 4, 23, 3],
        'output_dim': 512,
        'heads': vision_heads,
        'width': vision_width,
        'input_resolution': 224
    }
    default.update(**kwargs)
    return ModifiedResNet(**default)


@register_model
def modified_resnet_50x4(**kwargs):
    vision_width = 80
    vision_heads = vision_width * 32 // 64
    default = {
        'layers': [4, 6, 10, 6],
        'output_dim': 640,
        'heads': vision_heads,
        'width': vision_width,
        'input_resolution': 288
    }
    default.update(**kwargs)
    return ModifiedResNet(**default)


@register_model
def modified_resnet_50x16(**kwargs):
    vision_width = 96
    vision_heads = vision_width * 32 // 64
    default = {
        'layers': [6, 8, 18, 8],
        'output_dim': 768,
        'heads': vision_heads,
        'width': vision_width,
        'input_resolution': 384
    }
    default.update(**kwargs)
    return ModifiedResNet(**default)


@register_model
def modified_resnet_50x64(**kwargs):
    vision_width = 128
    vision_heads = vision_width * 32 // 64
    default = {
        'layers': [3, 15, 36, 10],
        'output_dim': 1024,
        'heads': vision_heads,
        'width': vision_width,
        'input_resolution': 448
    }
    default.update(**kwargs)
    return ModifiedResNet(**default)
