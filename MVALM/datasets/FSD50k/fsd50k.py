from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, List, Callable, Dict

import pandas as pd

from ..base import AudioDataset


class FSD50k(AudioDataset):
    LABELS = {'multilabel': {'train': 'annot_train.parquet',
                             'val': 'annot_val.parquet',
                             'test': 'annot_test.parquet'}
              }
    DATA_DIR = {'multilabel': 'data'}
    LABEL_COLUMN = 'labels'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: int = None,
                 mono: bool = True,
                 length: Optional[int] = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'multilabel',
                 verbose: bool = False,
                 ):
        super().__init__(root=root, datasets_root=datasets_root, sample_rate=sample_rate, mono=mono, length=length,
                         transform=transform, target_transform=target_transform, split=split, label_type=label_type,
                         verbose=verbose)

    @property
    def num_classes(self) -> int:
        return len(self._load_label_names())

    @lru_cache
    def _load_label_names(self) -> Dict:
        return pd.read_parquet(self.root / 'idx_2_label.parquet').to_dict()['label']

    def idx2label(self, idx: int) -> str:
        return self._load_label_names()[idx]
