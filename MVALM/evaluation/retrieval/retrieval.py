from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from MVALM.datasets.outputs import DataloaderType
from MVALM.datasets.base import AudioCaptionDataset
from .retrieval_func import _create_classifier, _evaluate_retrieval, recall, create_ret_target


def create_dataloader(dataset, batch_size: int, num_workers: int = 0,
                      vision_attr=None,
                      vision_processor=None) -> DataLoader:
    def collate(batch):
        # if it's a dict, use get, else (namedtupl) use getattr
        getter = lambda obj, attr: obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr)

        audio = torch.stack([getter(b, 'audio') for b in batch])
        input_ids = torch.stack([getter(b, 'caption')['input_ids'] for b in batch])
        attention_mask = torch.stack([getter(b, 'caption')['attention_mask'] for b in batch])

        if vision_attr is not None:
            vision = torch.stack([vision_processor(getter(b, va)) for b in batch for va in vision_attr])
            return dict(audio=audio), dict(input_ids=input_ids, attention_mask=attention_mask), dict(image=vision)

        return dict(audio=audio), dict(input_ids=input_ids, attention_mask=attention_mask)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=num_workers, shuffle=False)


def retrieval(dataloader: DataloaderType,
              query: str,
              document: str,
              model: torch.nn.Module,
              reverse: bool = True,
              dataset_name: str = None,
              device: str = 'cuda',
              dtype: torch.dtype = torch.float16,
              nr_img=1,
              verbose: bool = False) -> List[Dict]:
    """
    Perform retrieval evaluation on a given dataset.

    I.E. audio-to-text (audio -> text) means that the audio modality is used as query and the text modality is used as
    document. The query is used to retrieve the document.
    """
    batch_idx = dict(audio=0, text=1, vision=2)
    assert query in batch_idx.keys(), f'query must be one of {batch_idx.keys()}'
    assert document in batch_idx.keys(), f'document must be one of {batch_idx.keys()}'

    dataset_name = getattr(dataloader, 'name', dataset_name)
    if dataset_name is None:
        try:
            dataset_name = dataloader.dataset.name
        except AttributeError:
            raise AttributeError('Dataset has no name, please provide one.')

    forwards = dict(
        text=lambda x: model(**x).t_proj,
        audio=lambda x: model(**x).a_proj,
        vision=lambda x: model(**x).i_proj
    )

    samples = len(dataloader.dataset)
    num_samples = dict(audio=np.ones(samples, dtype=int),
                       vision=np.zeros(samples, dtype=int) + nr_img,
                       text=dataloader.dataset.annot.captions.apply(len).values)
    num_query = num_samples[query]
    num_doc = num_samples[document]

    query_to_document = create_ret_target(num_query, num_doc)

    if verbose:
        print(f'Retrieval on {dataset_name}')

    out = []
    # Create the documents to be retrieved
    clf = _create_classifier(forward_func=forwards[document],
                             batch_idx=batch_idx[document],
                             dataloader=dataloader,
                             device=device, dtype=dtype, verbose=verbose)
    # Evaluate the query
    logits = _evaluate_retrieval(forward_func=forwards[query],
                                 classifier=clf,
                                 batch_idx=batch_idx[query],
                                 dataloader=dataloader,
                                 device=device, dtype=dtype, verbose=verbose)

    result = recall(logits, query_to_document)
    if verbose:
        res_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.items()])
        print(f"{query} -> {document}:", res_str)
    result['dataset'] = dataset_name
    result['query'] = query
    result['document'] = document
    result['setting'] = f'{query} -> {document}'
    out.append(result)

    if reverse:
        document_to_query = create_ret_target(num_doc, num_query)

        result_rev = recall(logits.T, document_to_query)
        if verbose:
            res_str = ", ".join([f"{k}: {v:.3f}" for k, v in result_rev.items()])
            print(f"{document} -> {query}:", res_str)
        result_rev['dataset'] = dataset_name
        result_rev['query'] = document
        result_rev['setting'] = f'{document} -> {query}'
        result_rev['document'] = query
        out.append(result_rev)

    return out
