from functools import lru_cache
from pathlib import Path
from typing import Union, List, Optional, Dict, Callable

import pandas as pd

from ..base import VideoDataset, AudioDataset


class VGGSound(VideoDataset):
    LABELS = {'classification': {'train': 'annot_train.parquet',
                                 'test': 'annot_test.parquet'},
              }

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 labels_type: str = 'classification',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose, label_type=labels_type)

    @property
    def num_classes(self) -> int:
        return len(self._load_label_names())

    @lru_cache
    def _load_label_names(self) -> Dict:
        return pd.read_parquet(self.root / 'idx_2_label.parquet').to_dict()['label']

    def idx2label(self, idx: int) -> str:
        return self._load_label_names()[idx]


class VGGSoundAudio(AudioDataset):
    LABELS = {'classification': {'test': 'annot_audio_test.parquet'},
              }

    DATA_DIR = {'classification': 'audio'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: Optional[int] = 41100,
                 mono: bool = True,
                 length: int = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'test',
                 label_type: str = 'classification',
                 verbose: bool = False):
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

    @property
    def name(self) -> str:
        return 'VGGSound'
