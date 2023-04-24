from pathlib import Path
from typing import Union, List, Optional, NamedTuple, Callable, Dict, Tuple

import pandas as pd
import torch
from PIL import Image

from ..utils import pd_read_any
from ..base import BaseDataset


class ImageOutput(NamedTuple):
    image: Union[Image.Image, torch.Tensor]
    label: Union[int, str, List]


class ImageDataset(BaseDataset):
    LABELS = {'classification': {'train': 'annot_train.parquet',
                                 'val': 'annot_val.parquet',
                                 'test': 'annot_test.parquet'}}
    DATA_DIR = {'classification': 'images'}

    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 transform: Callable = None,
                 target_transform: Callable = None,
                 verbose: bool = False,
                 label_type: str = 'classification',
                 indices: Optional[List] = None
                 ):
        super().__init__(root=root, datasets_root=datasets_root, split=split, label_type=label_type, verbose=verbose)
        self.transform = transform
        self.target_transform = target_transform
        self.indices = indices

        annot_path = [self.LABELS[self.label_type][split] for split in self.split]
        if None in annot_path:
            self.logger.warning("No annotation file found for the given split. This can be intentional if you are using"
                                "a folder dataset.")
        else:
            annot_path = [self.root / path for path in annot_path]
            self.annot = pd.concat([pd_read_any(path) for path in annot_path])

    @staticmethod
    def ffcv_fields() -> Dict:
        from ffcv.fields import RGBImageField, IntField
        return {'image': RGBImageField(write_mode='raw'), 'label': IntField()}

    def __len__(self) -> int:
        return self.annot.shape[0] if self.indices is None else len(self.indices)

    def __getitem__(self, idx) -> ImageOutput:
        if self.indices is not None:
            idx = self.indices[idx]

        dp = self.annot.iloc[idx]
        filename, label = dp['filename'], dp['target']

        image_path = self.root / self.DATA_DIR[self.label_type] / filename

        if not image_path.exists():
            raise ValueError(f"Image {image_path} not found.")

        image = Image.open(image_path).convert('RGB')
        if image is None:
            raise ValueError(f"Corrupted image found at {image_path}.")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if isinstance(label, list):
            self.logger.warning('The label is a list of values. Consider using target_transform to choose one value.')

        return ImageOutput(image=image, label=label)


class FolderDataset(ImageDataset):
    def __init__(self,
                 root: Optional[Union[Path, str]] = None,
                 datasets_root: Optional[Union[Path, str]] = None,
                 split: Union[str, List[str]] = 'train',
                 transform: Callable = None,
                 target_transform: Callable = None,
                 verbose: bool = False,
                 label_type: str = 'classification'):
        super().__init__(root=root, datasets_root=datasets_root, split=split,
                         transform=transform, target_transform=target_transform, verbose=verbose,
                         label_type=label_type)
        assert len(self.split) == 1, "FolderDataset only supports a single split."

        self.classes, self.class_to_idx = self.find_classes(self.root / self.split[0])
        samples = self.make_dataset(self.root / self.split[0], self.class_to_idx)

        self.annot = pd.DataFrame(samples, columns=['filename', 'target'])

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset. """

        classes = []
        for path in Path(directory).iterdir():
            if path.is_dir():
                name = path.name
                classes.append(name)

        classes = sorted(classes, key=lambda x: int(x) if x.isdigit() else x)

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(self, directory: Path, class_to_idx: Dict) -> List[Tuple[str, int]]:
        directory = directory.expanduser()

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys(), key=lambda x: int(x) if x.isdigit() else x):
            class_index = class_to_idx[target_class]
            target_dir = directory / target_class
            if not target_dir.is_dir():
                continue
            for fnames in sorted(list(target_dir.iterdir())):
                item = fnames, class_index
                instances.append(item)

            if target_class not in available_classes:
                available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)
        return instances
