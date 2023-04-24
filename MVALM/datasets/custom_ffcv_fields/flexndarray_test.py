from tempfile import NamedTemporaryFile

import numpy as np
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter

from MVALM.datasets.custom_ffcv_fields import FlexNDArrayField, FlexNDArrayDecoder


class DemoDataset:
    def __init__(self, size=100):
        self.size = size
        self.data = [np.random.randn(10, 512).astype('float32') for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx],


if __name__ == '__main__':
    with NamedTemporaryFile() as handle:
        dataset = DemoDataset()

        writer = DatasetWriter(handle.name, {
            'field1': FlexNDArrayField(dtype=np.dtype('float32'), fixed_dim=512)
        })
        writer.from_indexed_dataset(dataset, shuffle_indices=False)

        loader = Loader(handle.name,
                        batch_size=10,
                        num_workers=2,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={'field1': [FlexNDArrayDecoder(), ]},
                        custom_fields={'field1': FlexNDArrayField})

        for i, data in enumerate(loader):
            field1 = data[0]
            for j, dp in enumerate(field1):
                assert dp in dataset.data[i * loader.batch_size + j]
        print("Test successful.")
