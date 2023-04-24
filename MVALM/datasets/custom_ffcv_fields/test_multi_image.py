from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from ffcv.fields.rgb_image import imdecode, encode_jpeg
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from MVALM.datasets.custom_ffcv_fields import MultiRGBImageField, MultiRGBImageDecoder


class DemoDataset:
    def __init__(self, size=100):
        roots = [
            '/home/t9s9/Datasets/AudioCaps/images_frame_1/val',
            '/home/t9s9/Datasets/AudioCaps/images_frame_5/val',
            '/home/t9s9/Datasets/AudioCaps/images_frame_9/val',
        ]
        self.files = [
            list(Path(root).iterdir())[:size]
            for root in roots
        ]

        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        out = [Image.open(file[idx]).convert('RGB') for file in self.files]
        return (out,)


def encode_decode(image, height, width):
    as_jpg = encode_jpeg(np.array(image), 90)
    destination = np.zeros((height, width, 3), dtype=np.uint8)
    imdecode(as_jpg, destination, height, width, height, width, 0, 0, 1, 1, False, False)
    return destination


if __name__ == '__main__':
    with NamedTemporaryFile() as handle:
        dataset = DemoDataset(100)

        writer = DatasetWriter(handle.name, {
            'image': MultiRGBImageField(num_images=3, write_mode='jpg', max_resolution=224),
        })
        writer.from_indexed_dataset(dataset, shuffle_indices=False)

        loader = Loader(handle.name,
                        batch_size=16,
                        num_workers=4,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': [MultiRGBImageDecoder()]
                        },
                        custom_fields={
                            'image': MultiRGBImageField
                        })

        for i, data in enumerate(loader):
            field1 = data[0]
            for j, dp in enumerate(field1):
                correct = any(
                    [np.allclose(dp, encode_decode(gt, 224, 224)) for gt in dataset[i * loader.batch_size + j][0]])
                assert correct, f"Index {i * loader.batch_size + j}"
        print("Test successful.")
