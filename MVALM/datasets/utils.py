from itertools import repeat
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel
from tqdm.auto import tqdm


def to_ntuple(x, n: int):
    if isinstance(x, (list, tuple)):
        assert len(x) == n, "Input must have length 3."
        return x
    return tuple(repeat(x, n))


def to_3tuple(x):
    return to_ntuple(x, 3)


def multi_one_hot(labels: Union[torch.tensor, np.ndarray, List], num_classes: int, torch_tensor: bool = False) -> Union[
    torch.tensor, np.ndarray]:
    if isinstance(labels, list):
        labels = torch.tensor(labels) if torch_tensor else np.array(labels)
    if isinstance(labels, np.ndarray):
        return np.eye(num_classes, dtype=np.dtype('int'))[labels].sum(0)
    elif isinstance(labels, torch.Tensor):
        return torch.eye(num_classes, dtype=torch.dtype('int'))[labels].sum(0)


def pad_or_truncate(x: Union[torch.tensor, np.ndarray], max_length: int) -> Union[torch.tensor, np.ndarray]:
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            if len(x) <= max_length:
                return np.concatenate((x, np.zeros(max_length - len(x), dtype=np.float32)), axis=0)
            else:
                return x[0: max_length]
        else:
            if x.shape[-1] <= max_length:
                return np.concatenate((x, np.zeros((x.shape[0], max_length - x.shape[-1]), dtype=np.float32)), axis=1)
            else:
                return x[:, :max_length]
    elif isinstance(x, torch.Tensor):
        if x.ndim == 1:
            if len(x) <= max_length:
                return torch.cat((x, torch.zeros(max_length - len(x), dtype=torch.float32)), dim=0)
            else:
                return x[0: max_length]
        else:
            if x.shape[-1] <= max_length:
                return torch.cat((x, torch.zeros((x.shape[0], max_length - x.shape[-1]), dtype=torch.float32)), dim=1)
            else:
                return x[:, :max_length]


def pd_read_any(path: Union[Path, str]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f'Unknown file type: {path}')


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class RecursivelyIterDir:
    def __init__(self, dir):
        self.base = Path(dir)
        self.iter = self.base.rglob('*')

    def __next__(self):
        path = self.iter.__next__()
        if path.is_file():
            return path
        else:
            return self.__next__()

    def __iter__(self):
        return self
