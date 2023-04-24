from pathlib import Path
from typing import Union, List, Optional

from ..base import VideoCaptionDataset


class MSRVTT(VideoCaptionDataset):
    LABELS = {'caption': {'train': 'annot_train.csv',
                          'val': 'annot_val.csv',
                          'test': 'annot_test.csv'},
              'retrieval': {'test': 'retrieval_train.csv'},
              }

    def __int__(self,
                root: Optional[Union[Path, str]] = None,
                datasets_root: Optional[Union[Path, str]] = None,
                split: Union[str, List[str]] = 'train',
                verbose: bool = False,
                **kwargs):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose)
