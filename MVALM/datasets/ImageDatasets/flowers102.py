from pathlib import Path
from typing import Union, List, Optional, Callable

from .base import ImageDataset


class Flowers102(ImageDataset):
    num_classes = 102

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 transform: Callable = None,
                 target_transform: Callable = None,
                 verbose: bool = False,
                 label_type: str = 'classification',
                 indices: Optional[List] = None):
        super().__init__(root=root, datasets_root=datasets_root, split=split, transform=transform,
                         target_transform=target_transform, verbose=verbose, label_type=label_type, indices=indices)
