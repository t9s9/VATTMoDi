from pathlib import Path

from ffcv.writer import DatasetWriter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize, CenterCrop, Compose

from MVALM.datasets.ImageDatasets import ImageNetV2, ImageNet, OxfordPet, Flowers102
from MVALM.datasets.ImageDatasets.base import ImageDataset


def create_beton(dataset: ImageDataset, output_dir: str):
    transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                         CenterCrop(224)])
    dataset.transform = transform

    output_path = str(Path(output_dir) / f'{dataset.name}_{dataset.split[0]}.beton')
    writer = DatasetWriter(output_path, dataset.ffcv_fields(), num_workers=16)
    writer.from_indexed_dataset(dataset, shuffle_indices=False)


def main():
    datasets = [
        Flowers102(datasets_root='/data/mmssl', split='test'),
        OxfordPet(datasets_root='/data/mmssl', split='test'),
        ImageNet(root='/data/ILSVRC12', split='val'),
        ImageNetV2(datasets_root='/data/mmssl', split='test'),
    ]
    for ds in datasets:
        print(ds)
        create_beton(ds, '/data/mmssl/beton/evaluation')
