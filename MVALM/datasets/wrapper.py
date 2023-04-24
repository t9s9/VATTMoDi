from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Callable, Union, Dict, Optional
from itertools import chain

import pandas as pd
from PIL import Image


def get_source_load_func(suffix: str) -> Callable:
    import torch, numpy, librosa

    if suffix == '.pt':
        return torch.load
    elif suffix == '.npy':
        return numpy.load
    elif suffix in ['.wav', '.flac']:
        return lambda f: librosa.load(f, sr=None, mono=True)
    elif suffix == '.mp3':
        return lambda f: numpy.fromfile(f, dtype='uint8')
    elif suffix in ['.jpg', '.png']:
        return lambda f: Image.open(f).convert('RGB')
    else:
        raise ValueError(f"Unknown source type: {suffix}")


class DatasetDataWrapper:
    """
    Wraps a dataset and adds additional data sources to it.

    This datasource can be used to add additional data to the dataset, such as audio, text, etc.
    It is represented as directories in the dataset root, where each directory contains the data for a single source.
    """

    def __init__(self,
                 dataset,
                 *sources: Union[str, Tuple[str, Callable]],
                 include_dataset_output: bool = False,
                 include_dataset_columns: Optional[Tuple[str, ...]] = None,
                 extra_annot_file: str = None,
                 ):
        self.dataset = dataset
        self.include_dataset_output = include_dataset_output
        self.include_dataset_columns = (
            {'filename', self.dataset.LABEL_COLUMN} if include_dataset_columns is None else set(include_dataset_columns)
        )
        self.extra_annot_file = extra_annot_file
        if self.extra_annot_file is not None:
            extra_annot = pd.concat(
                [pd.read_parquet(self.dataset.root / self.extra_annot_file.format(split=split)) for split in
                 self.dataset.split]
            )
            prev_len = len(self.dataset.annot)
            self.dataset.annot = pd.merge(self.dataset.annot, extra_annot, on='filename', how='inner', suffixes=('', '_y'))
            if len(self.dataset.annot) != prev_len:
                self.dataset.logger.warning(f"Extra annotation file must contain all samples. "
                                            f"Lost {prev_len - len(self.dataset.annot)} samples")
            self.dataset.annot = self.dataset.annot.reset_index(drop=True)
            self.include_dataset_columns = self.include_dataset_columns.union(extra_annot.columns)

        assert self.include_dataset_columns.issubset(self.dataset.annot.columns), \
            f"Dataset columns {self.include_dataset_columns} not found in dataset columns {self.dataset.annot.columns}"
        # assert len(sources) > 0, "At least one source must be provided"

        self.sources = [self.resolve_source(source) for source in sources]

    @property
    def source_names(self):
        return [source[0] for source in self.sources]

    def resolve_source(self, source: Union[str, Tuple[str, Callable]]) -> Tuple[str, Path, str, Callable]:
        """

        Args:
            source: Name of the source. It refers to a folder in the dataset root.

        Returns:
            (source_name, source_root, source_suffix, source_load_func)

        """
        if isinstance(source, str):
            source_name, source_load_func = source, None
        else:
            source_name, source_load_func = source

        source_root = self.dataset.root / source

        if not source_root.exists():
            raise ValueError(f"Source '{source} 'not found in '{self.dataset.root}'")

        for split in self.dataset.split:
            if not (source_root / split).exists():
                raise ValueError(f"Source '{source}' has no split '{split}' in {self.dataset.root}")

        source_files = [map(lambda x: x.parent.name + '/' + x.stem, (source_root / split).iterdir())
                        for split in self.dataset.split]
        source_files = set(chain(*source_files))

        exists = self.dataset.annot.filename.apply(lambda x: '.'.join(x.split('.')[:-1]) in source_files)
        self.dataset.logger.info(f"[{source_name}]: matching {exists.sum()}/{len(source_files)}")
        if exists.sum() < len(source_files):
            self.dataset.logger.warning(f"[{source_name}]: Could not assign all files from source to dataset.")

        self.dataset.logger.info(f"[{source_name}]: loosing {len(self.dataset.annot) - exists.sum()} samples")
        self.dataset.annot = self.dataset.annot[exists]

        # infer type of source
        source_suffix = next((source_root / self.dataset.split[0]).iterdir()).suffix

        self.dataset.logger.info(f"Found {source_name} source with suffix {source_suffix}")

        return source_name, source_root, source_suffix, source_load_func or get_source_load_func(source_suffix)

    def __str__(self):
        return f"DatasetDataWrapper({self.dataset}, {self.sources})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def __getitem__(self, index: int) -> Dict:
        output = {}
        dp = self.dataset.get_annot(index)

        for source_name, source_root, source_suffix, source_load_func in self.sources:
            source_file = (source_root / dp.filename).with_suffix(source_suffix)
            output[source_name] = source_load_func(source_file)

        if self.include_dataset_output:
            original = self.dataset[index]
            output.update(asdict(original))
        else:
            for col in self.include_dataset_columns:
                output[col] = dp[col]

        return output
