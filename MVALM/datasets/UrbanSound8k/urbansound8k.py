from pathlib import Path
from typing import Optional, Union, List, Callable

from ..base import AudioDataset


class UrbanSound8k(AudioDataset):
    LABELS = {'classification': {'test': 'urbansound.parquet'}}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: int = 44100,
                 mono: bool = True,
                 length: Optional[int] = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'test',
                 label_type: str = 'classification',
                 verbose: bool = False
                 ):
        super().__init__(root=root, datasets_root=datasets_root, sample_rate=sample_rate, mono=mono, length=length,
                         transform=transform, target_transform=target_transform, split=split, label_type=label_type,
                         verbose=verbose)
