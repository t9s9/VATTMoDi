from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import randperm
from torch.utils.data import DataLoader

from MVALM.datasets.utils import to_3tuple


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self,
                 batch_size: Union[int, List[int]],
                 num_workers: Union[int, List[int]],
                 root: Union[Path, str],
                 train_transform=None,
                 test_transform=None,
                 target_transform=None,
                 collate_fn: Union[Callable, List[Callable]] = None,
                 pin_memory: Union[bool, List[bool]] = True,
                 drop_last: Union[bool, List[bool]] = True,
                 ):
        super().__init__()
        self.root = Path(root)

        self.batch_size = to_3tuple(batch_size)
        self.num_workers = to_3tuple(num_workers)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.collate_fn = to_3tuple(collate_fn)
        self.pin_memory = to_3tuple(pin_memory)
        self.drop_last = to_3tuple(drop_last)
        self.sampler = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @staticmethod
    def random_split_idx(total, lengths: List) -> List[List]:
        assert sum(lengths) == total, f"Total and lengths need to be the same."

        generator = torch.Generator().manual_seed(42)
        indices = randperm(total, generator=generator).tolist()

        return [indices[offset - length: offset] for offset, length in zip(np.array(lengths).cumsum(), lengths)]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          batch_size=self.batch_size[0],
                          shuffle=True,
                          num_workers=self.num_workers[0],
                          pin_memory=self.pin_memory[0],
                          drop_last=self.drop_last[0],
                          collate_fn=self.collate_fn[0],
                          sampler=self.sampler)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          batch_size=self.batch_size[1],
                          shuffle=False,
                          num_workers=self.num_workers[1],
                          pin_memory=self.pin_memory[1],
                          drop_last=self.drop_last[1],
                          collate_fn=self.collate_fn[1],
                          sampler=self.sampler)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set,
                          batch_size=self.batch_size[2],
                          shuffle=False,
                          num_workers=self.num_workers[2],
                          pin_memory=self.pin_memory[2],
                          drop_last=self.drop_last[2],
                          collate_fn=self.collate_fn[2],
                          sampler=self.sampler)
