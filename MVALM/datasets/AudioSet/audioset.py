from functools import lru_cache
from pathlib import Path
from typing import Union, List, Optional, Callable, Dict

import pandas as pd

from ..base import VideoDataset, AudioDataset


class AudioSet(VideoDataset):
    LABELS = {'classification': {'val': 'annot_val.parquet',
                                 'train': 'annot_train.parquet'}}
    LABEL_COLUMN = 'labels'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type='classification',
                 verbose: bool = False,
                 limit=None):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose, label_type=label_type,
                         limit=limit)

    @property
    def num_classes(self) -> int:
        return 527

    @lru_cache
    def _load_label_names(self) -> Dict:
        return pd.read_parquet(self.root / 'class_labels_indices.parquet').to_dict()

    def idx2label(self, idx: int) -> str:
        return self._load_label_names()[idx]


class AudioSetAudioOnly(AudioDataset):
    LABELS = {'caption': {'train': 'annot_train.parquet',
                          'val': 'annot_val.parquet',
                          'test': 'annot_test.parquet'}
              }

    DATA_DIR = {'caption': 'data_audio'}

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
                 ):
        super().__init__(root=root, datasets_root=datasets_root, sample_rate=sample_rate, mono=mono, length=length,
                         transform=transform, target_transform=target_transform, split=split, label_type=label_type,
                         verbose=verbose)
        self.annot.filename = self.annot.filename.apply(lambda x: x.replace('mp4', 'flac'))

    @property
    def name(self) -> str:
        return 'AudioSet'

    @property
    def num_classes(self) -> int:
        return 527