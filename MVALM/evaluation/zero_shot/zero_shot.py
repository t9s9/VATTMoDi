import gc
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader, Dataset

from .zero_shot_data import get_templates, get_classes
from .zero_shot_func import _create_zero_shot_classifier, _evaluate_zero_shot


def zero_shot_eval(modality_forward,
                   text_forward,
                   dataloader,
                   tokenizer=None,
                   dataset_name: str = None,
                   batch_size: int = 64,
                   top_k: Tuple[int, ...] = (1, 2, 5, 10),
                   accuracy_average: str = 'micro',  # micro, macro, classwise
                   confusion_matrix: bool = False,
                   multi_label: bool = False,
                   simple_token: bool = False,
                   classifier: torch.Tensor = None,
                   device: str = 'cuda',
                   dtype: torch.dtype = torch.float32,
                   classes: List = None,
                   templates: List = None,
                   verbose: bool = True) -> Dict:
    dataset_name = getattr(dataloader, 'name', dataset_name)
    if dataset_name is None:
        try:
            dataset_name = dataloader.dataset.name
        except AttributeError:
            raise AttributeError('Dataset has no name, please provide one.')

    if accuracy_average == 'classwise':
        accuracy_average = None

    if classifier is None:
        if verbose:
            print("Creating classifier from templates")
        classes = get_classes(dataset_name) if classes is None else classes
        templates = get_templates(dataset_name, simple=simple_token) if templates is None else templates
        classifier = _create_zero_shot_classifier(forward_func=text_forward,
                                                  classnames=classes,
                                                  templates=templates,
                                                  tokenizer=tokenizer,
                                                  batch_size=batch_size,
                                                  device=device,
                                                  verbose=verbose)
    else:
        if verbose:
            print("Using given classifier.")
    classifier = classifier.to(device)
    result = _evaluate_zero_shot(forward_func=modality_forward, classifier=classifier, dataloader=dataloader,
                                 top_k=top_k, average=accuracy_average, device=device, dtype=dtype, verbose=verbose,
                                 confusion_matrix=confusion_matrix, multi_label=multi_label)
    if accuracy_average is None:  # classwise
        print(f"Zero-Shot-Result on {dataset_name}:")
        for k, v in result.items():
            if k == 'ConfusionMatrix':
                continue
            print(f"\t{k}:")
            for i, cls in enumerate(classes):
                print(f"\t\t{cls}: {v[i]:.3f}")
    else:
        res_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.items() if k != 'ConfusionMatrix'])
        print(f"Zero-Shot-Result on {dataset_name}: {res_str}")

    del classifier
    gc.collect()
    return result
