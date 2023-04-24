from pathlib import Path
from typing import Optional, List, Union, Callable

from .base import FolderDataset


class ImageNetV2(FolderDataset):
    LABELS = {'classification': {'test': None}}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'test',
                 transform: Callable = None,
                 target_transform: Callable = None,
                 verbose: bool = False,
                 label_type: str = 'classification'):
        super().__init__(root=root, datasets_root=datasets_root, split=split, transform=transform,
                         target_transform=target_transform, verbose=verbose, label_type=label_type)
