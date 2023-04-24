from typing import Callable, Union, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from MVALM.datasets.outputs import DataloaderType


def create_ret_target(num_query: List[int], num_doc: List[int]) -> Dict[int, List[int]]:
    """
    Creates a dictionary that maps query indices to document indices.
    :param num_query: Number of queries per document
    :param num_doc: Number of documents per query
    :return: Dictionary that maps query indices to document indices
    """
    assert len(num_query) == len(num_doc), 'Number of queries and documents does not match'

    query_to_document = {}
    q_counter, d_counter = 0, 0
    for q, d in zip(num_query, num_doc):
        for _ in range(q):
            query_to_document[q_counter] = list(range(d_counter, d_counter + d))
            q_counter += 1
        d_counter += d

    assert len(query_to_document) == sum(num_query)
    return query_to_document


def recall(logits: np.ndarray, modality_to_target: Dict) -> Dict:
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
    return {'R@1': tr1,
            'R@5': tr5,
            'R@10': tr10,
            'Mean': tr_mean}


def _create_classifier(forward_func: Callable,
                       batch_idx: int,
                       dataloader: DataloaderType,
                       device: str = "cuda",
                       dtype: torch.dtype = torch.float16,
                       verbose: bool = False) -> torch.Tensor:
    with torch.no_grad():
        weights = []
        bar = tqdm(dataloader, desc="Build query...", total=len(dataloader)) if verbose else dataloader
        for batch in bar:
            embeddings = batch[batch_idx]
            if isinstance(embeddings, dict):
                if 'image' in embeddings.keys():
                    embeddings = {k: v.to(device) for k, v in embeddings.items()}
                else:
                    embeddings = {k: v.reshape(-1, v.shape[-1]).to(device) for k, v in embeddings.items()}
            else:
                embeddings = embeddings.reshape(-1, embeddings.shape[-1]).to(device)

            # print(embeddings)

            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(embeddings)  # batch_size * num_tokens x embedding_dim
            features /= features.norm(dim=-1, keepdim=True)
            weights.append(features)
        weights = torch.concat(weights, dim=0).T

    return weights.to(device)


def _evaluate_retrieval(forward_func: Callable,
                        classifier: torch.Tensor,
                        batch_idx: int,
                        dataloader: DataloaderType,
                        device: str = "cuda",
                        dtype: torch.dtype = torch.float16,
                        verbose: bool = False) -> np.ndarray:
    matching = []
    with torch.no_grad():
        bar = tqdm(dataloader, desc=f'Predicting...', total=len(dataloader)) if verbose else dataloader
        for batch in bar:
            embeddings = batch[batch_idx]
            if isinstance(embeddings, dict):
                if 'image' in embeddings.keys():
                    embeddings = {k: v.to(device) for k, v in embeddings.items()}
                else:
                    embeddings = {k: v.reshape(-1, v.shape[-1]).to(device) for k, v in embeddings.items()}
            else:
                embeddings = embeddings.reshape(-1, embeddings.shape[-1]).to(device)

            # predict
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                features = forward_func(embeddings)

            features /= features.norm(dim=-1, keepdim=True)
            logits = features @ classifier
            matching.append(logits.cpu().numpy())

    return np.concatenate(matching)
