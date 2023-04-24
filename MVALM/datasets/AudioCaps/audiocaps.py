from pathlib import Path
from typing import Union, List, Optional, Callable

from ..base import VideoCaptionDataset, AudioCaptionDataset


class AudioCaps(VideoCaptionDataset):
    LABELS = {'caption': {'train': 'annot_tags_train.parquet',
                          'val': 'annot_tags_val.parquet',
                          'test': 'annot_tags_test.parquet'}
              }

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose)


class AudioCapsAudioOnly(AudioCaptionDataset):
    LABELS = {'caption': {'train': 'annot_tags_train.parquet',
                          'val': 'annot_tags_val.parquet',
                          'test': 'annot_tags_test.parquet'}
              }

    DATA_DIR = {'caption': 'audio'}

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
        self.annot.filename = self.annot.filename.apply(lambda x: x.replace('mp4', 'flac'))

    @property
    def name(self) -> str:
        return 'AudioCaps'
