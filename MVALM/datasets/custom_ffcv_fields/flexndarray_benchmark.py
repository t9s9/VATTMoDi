import time
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
from ffcv.fields import NDArrayField
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader

from flexndarray import FlexNDArrayField


class DemoDataset:
    def __init__(self, size: int, shape: Tuple[int, ...], random=True):
        self.size = size
        self.shape = shape
        self.random = random

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.random:
            array = np.random.randn(np.random.randint(1, self.shape[0]), self.shape[1]).astype('float32')
        else:
            array = np.random.randn(*self.shape).astype('float32')
        return (array,)


def run(dl, iterations=1, name='FFCV'):
    for data in dl: pass
    start_time = time.time()
    for _ in range(iterations):
        for data in dl: pass
    print(f'{name} | Time per epoch: {(time.time() - start_time) / iterations:.5f}s')


def benchmark_flex_array(shape=(10, 512), size=10000, iterations=1):
    dataset = DemoDataset(size, shape)
    with NamedTemporaryFile() as handle:
        writer = DatasetWriter(handle.name, {
            'field1': FlexNDArrayField(dtype=np.dtype('float32'), fixed_dim=shape[1])
        })
        writer.from_indexed_dataset(dataset)

        loader = Loader(handle.name,
                        batch_size=10,
                        num_workers=2,
                        order=OrderOption.RANDOM,
                        drop_last=True,
                        os_cache=True,
                        custom_fields={'field1': FlexNDArrayField})
        run(loader, iterations=iterations, name='FFCV FlexNDArrayField')


def benchmark_ndarray(shape=(512,), size=10000, iterations=1):
    dataset = DemoDataset(size, shape, random=False)
    with NamedTemporaryFile() as handle:
        writer = DatasetWriter(handle.name, {
            'field1': NDArrayField(dtype=np.dtype('float32'), shape=shape)
        })
        writer.from_indexed_dataset(dataset)

        loader = Loader(handle.name,
                        batch_size=10,
                        num_workers=2,
                        order=OrderOption.RANDOM,
                        drop_last=True,
                        os_cache=True)
        run(loader, iterations=iterations, name='FFCV NDArray')


def benchmark_torch(shape=(10, 512), size=10000, iterations=1):
    dataset = DemoDataset(size, shape)

    def sample(x):
        return np.stack([i[0][np.random.randint(0, i[0].shape[0])] for i in x], axis=0)

    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=2,
                        drop_last=True,
                        shuffle=True,
                        pin_memory=True,
                        collate_fn=sample)

    run(loader, iterations=iterations, name='PyTorch')


if __name__ == '__main__':
    iterations = 10
    shape = (10, 512)
    size = 50000

    benchmark_flex_array(shape=shape, size=size, iterations=iterations)
    benchmark_ndarray(shape=(shape[-1],), size=size, iterations=iterations)
    benchmark_torch(shape=shape, size=size, iterations=iterations)

    # FFCV FlexNDArrayField | Time per epoch: 1.08354s
    # FFCV NDArray     | Time per epoch: 0.96765s
    # PyTorch          | Time per epoch: 4.32856s
