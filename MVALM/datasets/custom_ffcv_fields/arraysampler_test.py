from tempfile import NamedTemporaryFile

import numpy as np
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from ffcv.transforms import ModuleWrapper
from ffcv.fields.ndarray import NDArrayField, NDArrayDecoder

from MVALM.datasets.custom_ffcv_fields.arraysampler import ArraySampler


class DemoDataset:
    def __init__(self, size=100):
        self.size = size
        self.data = [np.random.randn(5, 2, 512).astype('float32') for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx],


if __name__ == '__main__':
    with NamedTemporaryFile() as handle:
        dataset = DemoDataset(100)

        writer = DatasetWriter(handle.name, {
            'field1': NDArrayField(dtype=np.dtype('float32'), shape=(5, 2, 512)),
        })
        writer.from_indexed_dataset(dataset, shuffle_indices=False)

        loader = Loader(handle.name,
                        batch_size=4,
                        num_workers=1,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={'field1': [NDArrayDecoder(), ArraySampler(dim=0)]})

        for i, data in enumerate(loader):
            field1 = data[0]
            for j, dp in enumerate(field1):
                correct = np.isin(dp, dataset.data[i * loader.batch_size + j])
                assert correct.all(), f"Index {i * loader.batch_size + j}"
        print("Test successful.")
