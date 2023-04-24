from typing import Tuple

import numpy as np
import torch
from torchmetrics import MetricCollection, Accuracy
from tqdm import tqdm


def retrieval(document, query, module, dataloader, device=None, verbose=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_idx = dict(image=1, text=2, audio=3)

    modality_to_target = {}
    batch = next(iter(dataloader))
    samples = len(dataloader) * batch[0].shape[0]
    num_query = batch[batch_idx[query]].shape[1]
    num_doc = batch[batch_idx[document]].shape[1]

    for i, s in enumerate(range(0, samples * num_doc, num_doc)):
        for d in range(num_doc):
            modality_to_target[s + d] = list(range(i * num_query, i * num_query + num_query))

    if document == "image" and query == 'text':
        query_forward = lambda x: module.head.project(text_embed=x).ti_proj
        doc_forward = lambda x: module.head.project(image_embed=x).it_proj
    elif document == 'text' and query == 'image':
        query_forward = lambda x: module.head.project(image_embed=x).it_proj
        doc_forward = lambda x: module.head.project(text_embed=x).ti_proj
    elif document == 'image' and query == 'audio':
        query_forward = lambda x: module.head.project(audio_embed=x).ai_proj
        doc_forward = lambda x: module.head.project(image_embed=x).ia_proj
    elif document == 'audio' and query == 'image':
        query_forward = lambda x: module.head.project(image_embed=x).ia_proj
        doc_forward = lambda x: module.head.project(audio_embed=x).ai_proj
    elif document == 'text' and query == 'audio':
        query_forward = lambda x: module.head.project(audio_embed=x).at_proj
        doc_forward = lambda x: module.head.project(text_embed=x).ta_proj
    elif document == 'audio' and query == 'text':
        query_forward = lambda x: module.head.project(text_embed=x).ta_proj
        doc_forward = lambda x: module.head.project(audio_embed=x).at_proj
    else:
        raise ValueError(f"{document} and {query} are not supported")

    classifier = _create_classifier(query_forward,
                                    dataloader=dataloader,
                                    batch_idx=batch_idx[query],
                                    device=device,
                                    verbose=verbose)

    logits = _evaluate_retrieval(forward_func=doc_forward,
                                 classifier=classifier,
                                 batch_idx=batch_idx[document],
                                 dataloader=dataloader,
                                 device=device,
                                 verbose=verbose)

    ranks = np.zeros(logits.shape[0])
    for index, score in enumerate(logits):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in modality_to_target[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    name = f'{document}->{query}'
    eval_result = {f'{name}_r1': tr1,
                   f'{name}_r5': tr5,
                   f'{name}_r10': tr10,
                   f'{name}_r_mean': tr_mean}
    return eval_result


def _create_classifier(forward_func,
                       batch_idx: int,
                       dataloader,
                       device: str = "cuda",
                       verbose: bool = False):
    with torch.no_grad():
        weights = []
        bar = tqdm(dataloader, desc="Build query...", total=len(dataloader)) if verbose else dataloader
        for batch in bar:
            embeddings = batch[batch_idx]
            embeddings = embeddings.reshape(-1, embeddings.shape[-1]).to(device)

            features = forward_func(embeddings)  # batch_size * num_tokens x embedding_dim
            features /= features.norm(dim=-1, keepdim=True)
            weights.append(features)
        weights = torch.concat(weights, dim=0).T
    return weights.to(device)


def _evaluate_retrieval(forward_func,
                        classifier,
                        batch_idx: int,
                        dataloader,
                        top_k: Tuple[int, ...] = (1, 2, 5, 10),
                        average: str = "micro",
                        device: str = "cuda",
                        verbose: bool = False):
    num_classes = classifier.shape[1]  # classifier shape (embed, num_classes)
    metric = MetricCollection(
        {f'Top{k}Accuracy': Accuracy(threshold=0.5, top_k=k, average=average, num_classes=num_classes) for k in top_k}
    )

    matching = []
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...', total=len(dataloader)) if verbose else dataloader
        for batch in bar:
            # i, image_embeddings, text_embeddings, audio_embeddings = batch
            embeddings = batch[batch_idx]
            embeddings = embeddings.reshape(-1, embeddings.shape[-1]).to(device)

            # predict
            features = forward_func(embeddings)

            features /= features.norm(dim=-1, keepdim=True)
            logits = features @ classifier
            matching.append(logits.cpu().numpy())

    return np.concatenate(matching)
