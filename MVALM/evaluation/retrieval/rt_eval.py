from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from MVALM.datasets import Clotho, AudioCapsAudioOnly, MACS
from MVALM.evaluation.retrieval.retrieval import retrieval, create_dataloader
from MVALM.evaluation.utils import load_model


@lru_cache
def get_ret_dl(dataset_name: str, module, batch_size=16, num_workers=4, add_images: Tuple[str] = None,
               datasets_root=None):
    kwargs = dict(mono=True,
                  sample_rate=32000,
                  length=320000,
                  target_transform=lambda s: module.model.text_encoder.tokenize(list(s),
                                                                                padding='max_length'))
    dl_kwargs = {}
    if dataset_name == 'AudioCaps':
        datasets_root = datasets_root or '/home/t9s9/Datasets/'
        ds = AudioCapsAudioOnly(split='val', datasets_root=datasets_root, **kwargs)
        if add_images and hasattr(module.model, 'vision_encoder'):
            ds = ds.add_source(*add_images, include_dataset_output=True)
            dl_kwargs = dict(vision_attr=add_images,
                             vision_processor=module.model.vision_encoder.get_preprocessor(is_training=False))
    elif dataset_name == 'Clotho':
        datasets_root = datasets_root or '/media/t9s9/SSD_ubuntu/datasets'
        ds = Clotho(split='test', datasets_root=datasets_root, **kwargs)
    elif dataset_name == 'MACS':
        datasets_root = datasets_root or '/media/t9s9/SSD_ubuntu/datasets'
        ds = MACS(split='val', datasets_root=datasets_root, **kwargs)
        batch_size = 1  # MACS has different number of captions for each audio
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    return create_dataloader(ds, batch_size=batch_size, num_workers=num_workers, **dl_kwargs)


def rt_run(ckpt: str,
           datasets: List[str],
           momentum=False,
           datasets_root=None,
           batch_size=16,
           device='cuda') -> pd.DataFrame:
    result = []
    module = load_model(ckpt, device)

    if isinstance(momentum, bool):
        momentum = [momentum]
    else:
        momentum = [False, True]
    if not hasattr(module, 'model_m'):
        momentum = [False]

    for dataset_name in datasets:
        dataloader = get_ret_dl(dataset_name, module, add_images=('images_frame_3',), datasets_root=datasets_root,
                                batch_size=batch_size)
        for mom in momentum:
            if mom:
                vat_module = module.model_m
            else:
                vat_module = module
            res = retrieval(dataloader, query='audio', document='text', model=vat_module, device=device, verbose=False,
                            reverse=True)
            [r.update({'momentum': mom, 'ckpt': Path(ckpt).name}) for r in res]
            result.extend(res)

            if dataset_name == 'AudioCaps' and hasattr(module.model, 'vision_encoder'):
                for q, d in [('audio', 'vision'), ('text', 'vision')]:
                    res = retrieval(dataloader, query=q, document=d, model=vat_module, device=device,
                                    verbose=False, reverse=True, nr_img=1)
                    [r.update({'momentum': mom, 'ckpt': Path(ckpt).name}) for r in res]
                    result.extend(res)

    return pd.DataFrame(result)
