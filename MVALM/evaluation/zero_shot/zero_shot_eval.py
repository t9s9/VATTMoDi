from pathlib import Path
from typing import List

import pandas as pd
import torch

from MVALM.evaluation.utils import load_model, eval_dataset
from MVALM.evaluation.zero_shot import zero_shot_eval

templates_collection = {
    'no': ['{}'],
    'simple': ['The sound of {}'],
    'ensemble': [
        'A sound of {}',
        'It sounds like {}',
        'Audio of {}',
        'This is a sound of {}',
        'I can hear {}',
        'This is a sound of {}',
        'An audio clip of {}',
    ]
}


def zs_run(datasets: List[str],
           templates: List[List[str]],
           module=None,
           ckpt=None,
           momentum=False,
           confusion_matrix=True,
           device='cuda') -> pd.DataFrame:
    if module is None:
        module = load_model(ckpt, device)

    if isinstance(momentum, bool):
        momentum = [momentum]
    else:
        momentum = [False, True]
    if not hasattr(module, 'model_m'):
        momentum = [False]

    if isinstance(templates, dict):
        template_names = list(templates.keys())
        templates = list(templates.values())
    else:
        template_names = [str(i) for i in templates]

    df = []
    for ds in datasets:
        dl, multi_label = eval_dataset(ds, batch_size=8, num_workers=4)
        for mom in momentum:
            if mom:
                vat_module = module.model_m
            else:
                vat_module = module

            for temp_i, tmp in enumerate(templates):
                res = zero_shot_eval(
                    modality_forward=lambda x: vat_module(spectrogram=x).a_proj,
                    text_forward=lambda x: vat_module(input_ids=x.input_ids, attention_mask=x.attention_mask).t_proj,
                    dataloader=dl,
                    tokenizer=module.model.text_encoder.tokenize,
                    device=device,
                    top_k=(1, 5,),
                    dtype=torch.float16,
                    confusion_matrix=confusion_matrix,
                    templates=tmp,
                    multi_label=multi_label,
                    batch_size=8
                )
                res['template'] = template_names[temp_i]
                res['dataset'] = ds
                res['momentum'] = mom
                if ckpt is not None:
                    res['ckpt'] = Path(ckpt).name
                df.append(res)
                print("Memeory reserverd:", torch.cuda.memory_reserved() / 1e6)
                print("Memory allocated:", torch.cuda.memory_allocated() / 1e6)

    torch.cuda.empty_cache()
    return pd.DataFrame(df)
