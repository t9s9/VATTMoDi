import hashlib
import os
import urllib
import warnings
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from .text_encoder import base_text_transformers, large_text_transformers
from ..backbone import get_model

_MODELS = {
    "modified_resnet_50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "modified_resnet_101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "modified_resnet_50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "modified_resnet_50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "modified_resnet_50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "vit_base_patch32_224": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "vit_base_patch16_224": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "vit_large_patch14_224": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "vit_large_patch14_336": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


class CLIP(nn.Module):
    def __init__(self,
                 text_encoder,
                 image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def initialize_parameters(self):
        self.text_encoder.initialize_parameters()
        self.image_encoder.initialize_parameters()

    @classmethod
    def build(cls, visual_backbone_name: str, pretrained: bool = True, download_root: str = None,
              logit_scale: float = None, fp16: bool = False, **kwargs):
        image_encoder = get_model(visual_backbone_name)(**kwargs)
        text_encoder_kwargs = dict(modified_resnet_50=dict(embed_dim=1024),
                                   modified_resnet_101=dict(embed_dim=512),
                                   modified_resnet_50x4=dict(embed_dim=640, transformer_width=640,
                                                             transformer_heads=10),
                                   modified_resnet_50x16=dict(embed_dim=768, transformer_width=768,
                                                              transformer_heads=12),
                                   modified_resnet_50x64=dict(embed_dim=1024, transformer_width=1024,
                                                              transformer_heads=16),
                                   vit_base_patch16_224=dict(embed_dim=512),
                                   vit_base_patch32_224=dict(embed_dim=512),
                                   vit_large_patch14_224=dict(embed_dim=768, transformer_width=768,
                                                              transformer_heads=12, transformer_layers=12),
                                   vit_large_patch14_336=dict(embed_dim=768, transformer_width=768,
                                                              transformer_heads=12, transformer_layers=12)
                                   )
        text_encoder = base_text_transformers(**text_encoder_kwargs[visual_backbone_name], **kwargs)

        model = cls(text_encoder, image_encoder)
        if pretrained:
            state_dict = load_clip_state_dict(visual_backbone_name, download_root)
            # state_dict = convert_attention_loading(state_dict)

            if logit_scale is not None:
                state_dict['logit_scale'] = torch.tensor(logit_scale)
            model.load_state_dict(state_dict)
        else:
            model.initialize_parameters()
        if fp16:
            convert_weights(model)
        return model

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def encode_text(self, text):
        return self.text_encoder.encode_text(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def load_clip_state_dict(name, download_root: str = None) -> Dict:
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list(_MODELS.keys())}")

    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu")

    def clean(item):
        if item[0].startswith('visual.'):
            return 'image_encoder' + item[0][len('visual'):], item[1]
        elif item[0].startswith('transformer.') or item[0] in ["positional_embedding", "text_projection",
                                                               "token_embedding.weight", "ln_final.weight",
                                                               "ln_final.bias"]:
            return 'text_encoder.' + item[0], item[1]
        else:
            return item

    state_dict = dict(map(clean, state_dict.items()))
    delete_keys = ["input_resolution", "context_length", "vocab_size"]
    state_dict = dict(filter(lambda item: item[0] not in delete_keys, state_dict.items()))
    return state_dict


def convert_attention_loading(state_dict):
    def clean(item):
        convert = {'.attn.in_proj_weight': '.attn.qkv.weight',
                   '.attn.in_proj_bias': '.attn.qkv.bias',
                   '.attn.out_proj.weight': '.attn.proj.weight',
                   '.attn.out_proj.bias': '.attn.proj.bias'}
        new_key = item[0]
        for word, new_word in convert.items():
            new_key = new_key.replace(word, new_word)
        return new_key, item[1]

    return dict(map(clean, state_dict.items()))


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16. Inplace.
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def clip_rn50(pretrained: bool = True, download_root: str = None):
    return CLIP.build('modified_resnet_50', pretrained, download_root)


def clip_rn101(pretrained: bool = True, download_root: str = None):
    return CLIP.build('modified_resnet_101', pretrained, download_root)


def clip_vit_base_16(pretrained: bool = True, download_root: str = None):
    return CLIP.build('vit_base_patch16_224', pretrained, download_root)


def clip_vit_base_32(pretrained: bool = True, download_root: str = None):
    return CLIP.build('vit_base_patch32_224', pretrained, download_root)


def clip_vit_large(pretrained: bool = True, download_root: str = None):
    return CLIP.build('vit_large_patch14_224', pretrained, download_root)
