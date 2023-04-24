import io
import tarfile
from pathlib import Path
from typing import Optional, Union, List

from .base import BaseDataset, VideoCaptionOutput
from .video_utils import read_video_from_stream


class TarVideoCaptionDataset(BaseDataset):
    LABELS = {'video': {'train': 'annot_train.csv',
                        'val': 'annot_val.csv',
                        'test': 'annot_test.csv'},
              }
    DATA_DIR = {'video': 'data'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'video',
                 verbose: bool = False):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose)
        self.tar = tarfile.open(path, 'r')

        members = self.tar.getmembers()
        self.members = list(filter(lambda x: x.isfile(), members))

    def __len__(self):
        return len(self.members)

    def __del__(self):
        self.tar.close()

    def __getitem__(self, idx) -> VideoCaptionOutput:
        member = self.members[idx]

        stream = io.BytesIO(self.tar.extractfile(member).read())
        video, audio, meta = read_video_from_stream(stream)
        print(video.shape, audio.shape, meta)

        return
