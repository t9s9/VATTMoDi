from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union, Literal
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from MVALM.datasets import MSVD, AudioCaps, MSRVTT, VaTeX, VGGSound, ActivityNetDense
import MVALM.datasets as ds
from MVALM.datasets.FeatureDataset.extraction_dataset import ExtractionOutput
from MVALM.datasets.base_dataloader import BaseDataModule
from MVALM.models import CLIPTokenizer, CLIPImageProcessor
import re


def pre_caption(caption: str) -> str:
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption += '.'
    return caption


class ExtractionTransform:
    def __init__(self,
                 spectrogram_width: int = 1000,
                 train: bool = True,
                 prefix_caption_type_str: bool = False,
                 prefix_caption_type_token: bool = False,
                 ):
        self.train = train
        self.spectrogram_width = spectrogram_width
        self.prefix_caption_type_str = prefix_caption_type_str
        self.prefix_caption_type_token = prefix_caption_type_token

        # if self.prefix_caption_type_token:
        #     self.tokenizer.tokenizer.add_special_tokens(['<|visual|>', '<|auditory|>'])

        self.image_processor = CLIPImageProcessor(do_center_crop=False, random_flip=train)

    # def _tokenize(self, s: str, token_type: Optional[Literal['visual', 'auditory']] = None):
    #     special_token = None
    #     if self.prefix_caption_type_str:
    #         s = f"{token_type}: {s}"
    #     elif self.prefix_caption_type_token:
    #         special_token = '<|auditory|>' if token_type == 'auditory' else '<|visual|>'
    #
    #     return self.tokenizer(s, prefix_special_token=special_token).squeeze(0)

    def __call__(self, out: ExtractionOutput) -> Tuple[torch.Tensor, str, torch.Tensor]:
        # Pad or crop spectrogram
        spectrogram = out.spectrogram
        if spectrogram.shape[1] < self.spectrogram_width:
            zeros = np.zeros(shape=(128, self.spectrogram_width - spectrogram.shape[1]))
            spectrogram = np.concatenate([spectrogram, zeros], axis=1).astype('float32')
        else:
            spectrogram = spectrogram[:, :self.spectrogram_width].astype('float32')

        if self.train:
            # flip a coin
            if random.random() > 0.5:
                style = 'visual'
                caption = out.visual_captions
            else:
                style = 'auditory'
                caption = out.auditory_captions

            # select a caption
            caption = random.choice(caption)

            # Randomly select an image
            image = self.image_processor(random.choice(out.images))
        else:
            # use a random generator with seed to sample a caption
            if random.Random(42).random() > 0.5:
                style = 'visual'
                caption = out.visual_captions
            else:
                style = 'auditory'
                caption = out.auditory_captions

            caption = caption[0]

            # Select the middle image
            image = self.image_processor(out.images[len(out.images) // 2])

        return image, pre_caption(caption), torch.from_numpy(spectrogram)


class ExtractionDataModule(BaseDataModule):
    def __init__(self,
                 dataset_cls: List[str],
                 batch_size: Union[int, List[int]],
                 num_workers: Union[int, List[int]],
                 prefix_caption_type_str: bool = False,
                 prefix_caption_type_token: bool = False,
                 root: Union[Path, str] = None,
                 collate_fn: Union[Callable, List[Callable]] = None,
                 pin_memory: Union[bool, List[bool]] = True,
                 drop_last: Union[bool, List[bool]] = True):
        self.dataset_cls = dataset_cls
        train_transform = ExtractionTransform(train=True,
                                              prefix_caption_type_str=prefix_caption_type_str,
                                              prefix_caption_type_token=prefix_caption_type_token)
        test_transform = ExtractionTransform(train=False,
                                             prefix_caption_type_str=prefix_caption_type_str,
                                             prefix_caption_type_token=prefix_caption_type_token)

        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         root=root,
                         train_transform=train_transform,
                         test_transform=test_transform,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last)

    def setup(self, stage: Optional[str] = None):
        datasets = defaultdict(list)
        # available_datasets = [MSVD, AudioCaps, MSRVTT, VaTeX, VGGSound, ActivityNetDense]

        for dataset_cls in self.dataset_cls:
            dataset = getattr(ds, dataset_cls)
            labels_type = 'caption' if 'caption' in dataset.LABELS else 'classification'
            for split in dataset.LABELS[labels_type].keys():
                transform = self.train_transform if split == 'train' else self.test_transform
                print(f"Loading {dataset.__name__} ({split})")
                datasets[split].append(dataset.as_extraction_dataset(datasets_root=self.root,
                                                                     feature_dir='extraction-10s',
                                                                     split=split,
                                                                     number_images=3,
                                                                     image_pick_strategy='linspace',
                                                                     transform=transform))
        for split in datasets.keys():
            setattr(self, f"{split}_set", ConcatDataset(datasets[split]))
