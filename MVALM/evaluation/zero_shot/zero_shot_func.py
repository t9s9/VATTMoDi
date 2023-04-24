from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from .utils import make_batches
from ...models import CLIPTokenizer
from torchmetrics import MetricCollection, Accuracy
from torchmetrics.classification import MulticlassAccuracy, MultilabelAveragePrecision, MultilabelPrecision, \
    MultilabelRecall, MultilabelExactMatch, MulticlassConfusionMatrix


def _create_zero_shot_classifier(forward_func,
                                 classnames: Union[List, torch.Tensor, NDArray],
                                 templates: List = None,
                                 tokenizer=None,
                                 batch_size: int = 64,
                                 device: Union[str, torch.device] = "cuda",
                                 verbose: bool = False):
    tokenizer = CLIPTokenizer(context_length=77, truncate=True) if tokenizer is None else tokenizer
    templates = ['{}'] if templates is None else templates
    if isinstance(templates, str):
        templates = [templates]
    num_templates = len(templates)
    batch_size = 2 ** ((batch_size // num_templates) - 1).bit_length()

    do_tokenize = isinstance(classnames, (list, np.ndarray))
    batch_class_names = make_batches(classnames, batch_size)

    with torch.no_grad():
        zeroshot_weights = []
        bar = tqdm(batch_class_names, desc="Classifier weights...") if verbose else batch_class_names
        for batch_class_name in bar:
            if do_tokenize:
                texts = [template.format(classname) for classname in batch_class_name for template in
                         templates]  # format with class
                texts = tokenizer(texts).to(device)  # tokenize Shape: batch_size * num_tokens x context_length
            else:
                texts = classnames.to(device)

            class_embeddings = forward_func(texts)  # batch_size * num_tokens x embedding_dim

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embeddings = class_embeddings.view(len(batch_class_name), num_templates,
                                                     -1)  # batch_size x num_tokens x embedding_dim
            class_embedding_mean = class_embeddings.mean(dim=1)  # batch_size x embedding_dim
            class_embedding_mean /= class_embedding_mean.norm(dim=1).view(-1, 1)

            zeroshot_weights.append(class_embedding_mean)
        zeroshot_weights = torch.concat(zeroshot_weights, dim=0).T
    return zeroshot_weights.to(device)


def _evaluate_zero_shot(forward_func,
                        classifier,
                        dataloader,
                        top_k: Tuple[int, ...] = (1, 2, 5, 10),
                        average: str = "micro",
                        device: Union[str, torch.device] = "cuda",
                        dtype: torch.dtype = torch.float32,
                        confusion_matrix: bool = False,
                        multi_label: bool = False,
                        verbose: bool = False) -> Dict:
    num_classes = classifier.shape[1]  # classifier shape (embed, num_classes)

    if multi_label:
        metric = {
            f'mAP': MultilabelAveragePrecision(num_labels=num_classes, average='macro'),
        }
    else:
        metric = {
            f'Top{k}Accuracy': MulticlassAccuracy(threshold=0.5, top_k=k, average=average, num_classes=num_classes)
            for k in top_k
        }

    if confusion_matrix:
        metric['ConfusionMatrix'] = MulticlassConfusionMatrix(num_classes=num_classes, normalize=None)

    metric = MetricCollection(metric).to(device)

    classifier = classifier.to(dtype=dtype)
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...') if verbose else dataloader
        for point in bar:
            inputs, target = point
            inputs = inputs.to(device=device)
            target = target.to(device)

            # predict
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(inputs)

            features /= features.norm(dim=-1, keepdim=True)
            logits = features @ classifier

            step_metric = metric(logits, target.clone().squeeze().long())

            if verbose and average is not None:
                bar.set_postfix({k: v.item() for k, v in step_metric.items() if k != 'ConfusionMatrix'})

    result = metric.compute()
    for k, v in result.items():
        result[k] = v.cpu().numpy().item() if v.dim() == 0 else v.cpu().numpy()
    return result
