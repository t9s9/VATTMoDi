import io
import tarfile
from pathlib import Path
from typing import Union, List, Optional

from ..base import VideoCaptionOutput, VideoCaptionDataset
from ..video_utils import read_video_from_stream


class ActivityNetDense(VideoCaptionDataset):
    LABELS = {'caption': {'train': 'annot_train.parquet',
                          'val': 'annot_val_3000.parquet'},
              }
    DATA_DIR = {'caption': 'data'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose)


class ActivityNetDenseTar(VideoCaptionDataset):
    LABELS = {'video': {'train': 'annot_train.csv',
                        'val': 'annot_val.csv'}}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(root=root, datasets_root=datasets_root, split=split, verbose=verbose)
        self.members = []
        self.member_to_tar = []
        self.tar = [tarfile.open(self.root / 'video_clips' / f'{split}.tar', 'r') for split in self.split]

        for i, tar in enumerate(self.tar):
            members = list(filter(lambda x: x.isfile(), tar.getmembers()))
            self.members += members
            self.member_to_tar += [i] * len(members)

    def __len__(self):
        return len(self.members)

    def __del__(self):
        for tar in self.tar:
            tar.close()

    def __getitem__(self, idx) -> VideoCaptionOutput:
        member = self.members[idx]
        caption = self.annot[self.annot['filename'] == member.path].iloc[0]['captions']

        stream = io.BytesIO(self.tar[self.member_to_tar[idx]].extractfile(member).read())
        video_frames, audio_frames, meta = read_video_from_stream(stream)

        return VideoCaptionOutput(video=video_frames,
                                  audio=audio_frames,
                                  video_fps=meta.get('video_fps', None),
                                  audio_fps=meta.get('audio_fps', None),
                                  caption=caption,
                                  filename=str(member.path))
