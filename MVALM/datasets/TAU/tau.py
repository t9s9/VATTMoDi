from pathlib import Path
from typing import Optional, Union, List, Callable

import pandas as pd

from ..base import AudioDataset


class TAU(AudioDataset):
    LABELS = {'classification': {'test': 'annot.parquet'}}
    DATA_DIR = {'classification': 'audio'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: Optional[int] = 44100,
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
        self.id_to_category = self.annot[['target', 'category']].drop_duplicates().set_index('target').to_dict()[
            'category']
        self.annot.filename = self.annot.filename.apply(lambda x: str(Path(x).with_suffix('.flac')))

    def get_class_names(self) -> List[str]:
        _, class_names = zip(*sorted(self.id_to_category.items(), key=lambda x: x[0]))
        return list(map(lambda x: x.replace('_', ' ').strip(), class_names))
