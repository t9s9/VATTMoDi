from pathlib import Path
from typing import Union, List, Optional

from ..base import VideoCaptionDataset


class DiDeMo(VideoCaptionDataset):
    LABELS = {'caption': {'train': 'annot_clip_train.csv',
                          'val': 'annot_clip_val.csv',
                          'test': 'annot_clip_test.csv'},
              }
    DATA_DIR = {'video_clip': 'video_clips'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'caption',
                 verbose: bool = False):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose)
