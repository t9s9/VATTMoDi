from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from ffcv.loader import Loader
import seaborn as sns
import matplotlib.pyplot as plt
from MVALM.modules.vat_module import VATModule, ATModule
from MVALM.modules.vat_module_distill import VATModule as VATModuleDistill
from MVALM.modules.vat_module_distill import ATModule as ATModuleDistill
from MVALM.modules.vat_module_distill import DistillTrainingOptions

root = Path('')
EVAL_DATASETS = {
    'UrbanSound8k': root / 'UrbanSound8k-spec.beton',
    'ESC50': root / 'ESC50-spec.beton',
    'BDLib2': root / 'BDLib2-spec.beton',
    'VGGSound': root / 'VGGSound-spec.beton',
    'AudioSet': root / 'AudioSet-spec.beton',
    'FSD50k': root / 'FSD50kval-spec.beton',
    'TAU': root / 'TAU-spec.beton',
}


def draw_confusion_matrix(mat, classes, linewidth=0.5, annot=False):
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(data=mat, square=True, annot=annot, fmt='d', cbar=True, xticklabels=classes, yticklabels=classes,
                cmap="rocket", linewidth=linewidth, ax=ax).set(xlabel='Predicted label', ylabel='True label',
                                                               title='Confusion Matrix')
    return ax


def load_model(ckpt, device, half=True, mem_eff=False):
    sd = torch.load(ckpt)

    if 'loss_kwargs' in sd['hyper_parameters']:
        vat_cls, at_cls = VATModule, ATModule
        distill = False
    else:
        vat_cls, at_cls = VATModuleDistill, ATModuleDistill
        distill = True
        if "training_kwargs" not in sd['hyper_parameters']:
            sd['hyper_parameters']['training_kwargs'] = DistillTrainingOptions(**{
                k: sd['hyper_parameters'][k] for k in ['queue_size', 'temp', 'momentum', 'alpha']
            }, distill=True)

    sd['hyper_parameters']['model_kwargs']['text_encoder']['cache_dir'] = None

    if 'vision_encoder' in sd['hyper_parameters']['model_kwargs']:
        sd['hyper_parameters']['model_kwargs']['vision_encoder']['cache_dir'] = None
        vat_module = vat_cls.load_from_checkpoint(ckpt, **sd['hyper_parameters'], strict=False)
    else:
        vat_module = at_cls.load_from_checkpoint(ckpt, **sd['hyper_parameters'], strict=False)

    vat_module = vat_module.to(device).eval()
    if half:
        vat_module.half()
        vat_module.model.audio_encoder.spectrogram.float()
        if mem_eff:
            for block in vat_module.model.audio_encoder.model.blocks:
                block.attn.set_use_memory_efficient_attention_xformers(True)
        if distill:
            vat_module.model_m.audio_encoder.spectrogram.float()
            if mem_eff:
                for block in vat_module.model_m.audio_encoder.model.blocks:
                    block.attn.set_use_memory_efficient_attention_xformers(True)
    else:
        for block in vat_module.model.audio_encoder.model.blocks:
            block.attn.set_use_memory_efficient_attention_xformers(False)
        if distill:
            for block in vat_module.model_m.audio_encoder.model.blocks:
                block.attn.set_use_memory_efficient_attention_xformers(False)
    return vat_module


# @lru_cache
def eval_dataset(name, batch_size=32, num_workers=4) -> Tuple[Loader, bool]:
    multilabel = name in ['FSD50k', 'AudioSet']
    loader = Loader(EVAL_DATASETS[name], batch_size=batch_size, num_workers=num_workers)
    loader.name = name
    return loader, multilabel


def list_eval_datasets():
    return list(EVAL_DATASETS.keys())
