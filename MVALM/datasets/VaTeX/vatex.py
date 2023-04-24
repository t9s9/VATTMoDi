from pathlib import Path
from typing import Union, List, Optional

from ..base import VideoCaptionDataset


class VaTeX(VideoCaptionDataset):
    LABELS = {'caption': {'train': 'annot_train.parquet',
                          'val': 'annot_val.parquet',
                          'test': 'annot_test.parquet'},
              }

    def __int__(self,
                root: Optional[Union[Path, str]] = None,
                datasets_root: Optional[Union[Path, str]] = None,
                split: Union[str, List[str]] = 'train',
                verbose: bool = False):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose)
