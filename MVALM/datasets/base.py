import logging
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path
from typing import Union, List, Optional, Any, Callable, Iterable, Dict, Tuple

import pandas as pd
from librosa import load as load_audio
from torch import from_numpy
from torch.utils.data import Dataset
from torchvision.io import read_video

from .FeatureDataset.extraction_dataset import ExtractionDataset
from .outputs import AudioOutput, AudioCaptionOutput, VideoOutput, VideoCaptionOutput
from .utils import pd_read_any, bisect_right, pad_or_truncate
from .wrapper import DatasetDataWrapper


class BaseDataset(Dataset, ABC):
    LABELS = {'unsupervised': ['train', 'val', 'test']}
    DATA_DIR = {'unsupervised': 'data'}
    DEFAULT_ROOT = "/data/mmssl/"
    LABEL_COLUMN = 'target'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'unsupervised',
                 verbose: bool = False,
                 limit: Optional[int] = None,
                 ):
        super().__init__()
        self.verbose = verbose
        self.logger = self.init_logger()
        self.limit = limit

        self.root = self.resolve_path(root, datasets_root)
        self.label_type = self.resolve_label(label_type)
        self.split = self.resolve_split(split)

        self.logger.info('Initialized dataset {0}'.format(self.name))

        self.annot = self.create_annotation()

    def add_source(self, *sources,
                   include_dataset_output: bool = False,
                   include_dataset_columns: Optional[Tuple[str, ...]] = None,
                   **kwargs) -> DatasetDataWrapper:
        return DatasetDataWrapper(self, *sources, include_dataset_output=include_dataset_output,
                                  include_dataset_columns=include_dataset_columns, **kwargs)

    @classmethod
    def as_extraction_dataset(cls,
                              feature_dir: str,
                              root: Optional[Union[Path, str]] = None,
                              datasets_root: Optional[Union[Path, str]] = None,
                              **kwargs) -> ExtractionDataset:
        if root is not None:
            root = Path(root)
        elif datasets_root is not None:
            root = Path(datasets_root) / cls.__name__
        else:
            root = Path(cls.DEFAULT_ROOT) / cls.__name__

        return ExtractionDataset(root=root / feature_dir, **kwargs)

    def create_annotation(self) -> pd.DataFrame:
        annot_paths = [self.root / self.LABELS[self.label_type][split] for split in self.split]
        annot = pd.concat([pd_read_any(path) for path in annot_paths])

        if not {self.LABEL_COLUMN, 'filename'}.issubset(annot.columns):
            raise ValueError(
                f'Invalid annotation file. Found {annot.columns}. Expected: ({self.LABEL_COLUMN}, filename)')

        self.logger.info(f'Loaded annotation with {len(annot)} samples.')
        if self.limit is not None:
            annot = annot.iloc[:self.limit]
            self.logger.info(f'Limited to {len(annot)} samples.')
        return annot

    def init_logger(self):
        logger = logging.getLogger(f"{self.name}_logger")
        logger.setLevel(level=logging.INFO if self.verbose else logging.ERROR)
        if not logger.hasHandlers():
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level=logging.INFO if self.verbose else logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        return logger

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def available_splits(self) -> List[str]:
        return list(self.LABELS[self.label_type].keys())

    @property
    def available_labels(self) -> List[str]:
        return list(self.LABELS.keys())

    @property
    def data_dir(self) -> Path:
        return self.root / self.DATA_DIR[self.label_type]

    @classmethod
    def by_splits(cls, label_type: str = 'unsupervised', **kwargs) -> Dict[str, 'BaseDataset']:
        """ Returns a dictionary of datasets, one for each split. """
        return {split: cls(split=split, label_type=label_type, **kwargs) for split in cls.LABELS[label_type]}

    def __str__(self) -> str:
        return f"Dataset: {self.name}, Split: {','.join(self.split)}, Size: {len(self)}, Label: {self.label_type}"

    def __repr__(self) -> str:
        return f"{self.name}(split={';'.join(self.split)}, len={len(self)}, label={self.label_type})"

    def resolve_path(self, root: Union[Path, str], datasets_root: Union[Path, str]) -> Path:
        if root is None and datasets_root is None and self.DEFAULT_ROOT is None:
            raise ValueError("Specify the datasets path in `root` or `datasets_root`.")
        elif root is None and datasets_root is None:
            self.logger.warning(f"Using default dataset root dir: {self.DEFAULT_ROOT}.")
            return Path(self.DEFAULT_ROOT) / self.name
        elif root is None:
            return Path(datasets_root) / self.name
        return Path(root)

    def resolve_split(self, split: Union[str, List[str]]) -> List[str]:
        split_error = f'Unknown split {split}. Use one of {self.available_splits}, a combination of those or ' \
                      f'`all` to use all splits together.'
        if isinstance(split, str):
            split = split.lower()
            if split == 'all':
                split = self.available_splits
            elif split in self.available_splits:
                split = [split]
            else:
                raise ValueError(split_error)
        elif isinstance(split, list):
            for i, split_str in enumerate(split):
                if split_str.lower() in self.available_splits:
                    split[i] = split_str.lower()
                else:
                    self.logger.warning(f'Unknown split {split_str}. Continue searching for other valid split.')
                    del split[i]
            if not split:
                raise ValueError(split_error)
        else:
            raise ValueError(split_error)
        return split

    def resolve_label(self, label_type: str) -> str:
        if label_type not in self.available_labels:
            raise ValueError(f"Unknown label type {label_type}. Choose one of {', '.join(self.available_labels)}.")
        return label_type

    def verify_files(self, remove: bool = True) -> None:
        self.logger.info(f"Verifying files for {self.name}...")
        exists = self.annot.filename.apply(lambda x: (self.root / self.DATA_DIR[self.label_type] / x).exists())
        if not exists.all():
            self.logger.warning(f"Found {len(exists) - exists.sum()} missing files.")
            if remove:
                self.logger.warning("Removing missing files from the dataset.")
                self.annot = self.annot[exists]

    def __len__(self) -> int:
        return len(self.annot)

    def get_annot(self, index: int) -> pd.Series:
        return self.annot.iloc[index]

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class VideoDataset(BaseDataset):
    LABELS = {'classification': {'train': 'annot_train.parquet',
                                 'val': 'annot_val.parquet',
                                 'test': 'annot_test.parquet'},
              }
    DATA_DIR = {'classification': 'data'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'classification',
                 verbose: bool = False,
                 limit: Optional[int] = None):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose,
                         limit=limit)

    def __getitem__(self, index: int) -> VideoOutput:
        dp = self.get_annot(index)
        filename, label = dp['filename'], dp[self.LABEL_COLUMN]

        path = self.root / self.DATA_DIR[self.label_type] / filename
        video_frames, audio_frames, meta = read_video(str(path), pts_unit='sec')

        return VideoOutput(video=video_frames,
                           audio=audio_frames,
                           video_fps=meta.get('video_fps', None),
                           audio_fps=meta.get('audio_fps', None),
                           label=label,
                           filename=str(filename))


class VideoCaptionDataset(VideoDataset):
    LABELS = {'caption': {'train': 'annot_train.parquet',
                          'val': 'annot_val.parquet',
                          'test': 'annot_test.parquet'},
              }

    DATA_DIR = {'caption': 'data'}
    LABEL_COLUMN = 'captions'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'caption',
                 verbose: bool = False,
                 limit: Optional[int] = None):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose,
                         limit=limit)
        if isinstance(self.annot[self.LABEL_COLUMN].iloc[0], str):
            self.logger.warning("Captions are detected as strings. Converting them to lists.")
            self.annot.captions = self.annot.captions.apply(literal_eval)  # resolve list of captions

    def __getitem__(self, index: int) -> VideoCaptionOutput:
        out: VideoOutput = super().__getitem__(index)
        return VideoCaptionOutput(video=out.video, audio=out.audio, video_fps=out.video_fps, audio_fps=out.audio_fps,
                                  caption=out.label, filename=out.filename)


class AudioDataset(BaseDataset):
    LABELS = {'classification': {'train': 'annot_train.csv',
                                 'val': 'annot_val.csv',
                                 'test': 'annot_test.csv'},
              }
    DATA_DIR = {'classification': 'data'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: Optional[int] = 41100,
                 mono: bool = True,
                 length: int = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'classification',
                 verbose: bool = False):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose)
        self.sample_rate = sample_rate
        self.mono = mono
        self.length = length
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> AudioOutput:
        dp = self.get_annot(index)
        audio_path = self.data_dir / dp['filename']
        label = dp[self.LABEL_COLUMN]

        if not audio_path.exists():
            raise ValueError(f"Image {audio_path} not found.")

        audio, sr = load_audio(audio_path, mono=self.mono, sr=self.sample_rate)

        if self.length is not None:
            audio = pad_or_truncate(audio, self.length)

        if self.transform is not None:
            audio = from_numpy(audio)
            audio = audio.unsqueeze(0)
            audio = self.transform(audio)
        else:
            audio = from_numpy(audio)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return AudioOutput(audio=audio, audio_fps=sr, label=label, filename=dp['filename'])


class AudioCaptionDataset(AudioDataset):
    LABELS = {'captions': {'train': 'annot_train.csv',
                           'val': 'annot_val.csv',
                           'test': 'annot_test.csv'},
              }
    DATA_DIR = {'captions': 'data'}
    LABEL_COLUMN = 'captions'

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 sample_rate: Optional[int] = 41100,
                 mono: bool = True,
                 length: int = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 split: Union[str, List[str]] = 'train',
                 label_type: str = 'captions',
                 verbose: bool = False,
                 expand_captions: bool = False):
        super().__init__(root=root, datasets_root=datasets_root, sample_rate=sample_rate, mono=mono, length=length,
                         transform=transform, target_transform=target_transform, split=split, label_type=label_type,
                         verbose=verbose)
        if isinstance(self.annot[self.LABEL_COLUMN].iloc[0], str):
            # resolve list of captions
            self.logger.warning("Captions are detected as strings. Converting them to lists.")
            self.annot[self.LABEL_COLUMN] = self.annot[self.LABEL_COLUMN].apply(literal_eval)
        if expand_captions:
            self.annot = self.annot.explode(self.LABEL_COLUMN).reset_index(drop=True)

        self.has_tags = 'tags' in self.annot.columns


    def __getitem__(self, index: int) -> AudioCaptionOutput:
        out: AudioOutput = super().__getitem__(index)
        tags = self.annot.loc[index, 'tags'] if self.has_tags else None
        return AudioCaptionOutput(audio=out.audio, audio_fps=out.audio_fps, caption=out.label, filename=out.filename,
                                  tags=tags)


class ConcatDataset(Dataset):
    """
    Dataset as a concatenation of multiple datasets.

    Adapted from the torchvision implementation.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
        return_origin (bool): If True, return the name of the dataset from which the sample was taken
    """

    def __init__(self, datasets: Iterable[BaseDataset], return_origin: bool = True) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        # for ds in self.datasets:
        #     assert isinstance(ds, BaseDataset), 'datasets should be subclass of BaseDataset'
        self.return_origin = return_origin
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if self.return_origin:
            return self.datasets[dataset_idx][sample_idx], self.datasets[dataset_idx].name
        else:
            return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes

    @property
    def name(self):
        return 'ConcatDataset of ' + ' + '.join([ds.name for ds in self.datasets])

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
