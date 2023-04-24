from pathlib import Path
from typing import Optional, Union, List, Callable

import pandas as pd

from ..base import AudioCaptionDataset


class Clotho(AudioCaptionDataset):
    LABELS = {'caption': {'train': 'annot_train.parquet',
                          'val': 'annot_val.parquet',
                          'test': 'annot_test.parquet'}
              }
    DATA_DIR = {'caption': 'data'}
    LABEL_COLUMN = 'captions'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: int = None,
                 mono: bool = True,
                 length: Optional[int] = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'caption',
                 verbose: bool = False,
                 expand_captions: bool = False
                 ):
        super().__init__(root=root, datasets_root=datasets_root, sample_rate=sample_rate, mono=mono, length=length,
                         transform=transform, target_transform=target_transform, split=split, label_type=label_type,
                         verbose=verbose, expand_captions=expand_captions)
