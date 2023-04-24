from collections import namedtuple
from pathlib import Path

import torch

from .htsat import HTSAT_Swin_Transformer


def get_model_htsat(arch: str = 'htsat',
                    cache_dir: str = None,
                    pretrained: bool = True,
                    gradient_checkpointing: bool = True,
                    num_classes: int = 0,
                    **kwargs):
    audio_cfg = dict(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        enable_repeat_mode=False,
    )
    AudioCfg = namedtuple('AudioCfg', audio_cfg)
    audio_cfg = AudioCfg(**audio_cfg)

    model = HTSAT_Swin_Transformer(
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_head=[4, 8, 16, 32],
        window_size=8,
        use_checkpoint=gradient_checkpointing,
        config=audio_cfg,
    )

    if pretrained:
        filename = 'HTSAT_AudioSet_Saved_1.ckpt'
        if cache_dir is None:
            cache_dir = Path.home() / Path('.cache', 'torch', 'hub', 'checkpoints')
        else:
            cache_dir = Path(cache_dir)
        ckpt_path = cache_dir / filename
        if not ckpt_path.exists():
            print(f"Please download the checkpoint {filename} "
                  f"file from https://github.com/RetroCirce/HTS-Audio-Transformer")

        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd['state_dict']

        for key in list(sd.keys()):
            if key.startswith('sed_model.'):
                sd[key[10:]] = sd.pop(key)
        sd.pop('head.weight')  # the head is not used
        sd.pop('head.bias')
        if num_classes == 0:
            sd.pop('tscam_conv.weight')  # remove classifier
            sd.pop('tscam_conv.bias')

        model.load_state_dict(sd)

    return model
