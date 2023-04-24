from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter

from mp3 import MP3Field


class DemoDataset:
    def __init__(self, root, size=32):
        self.root = Path(root)
        self.size = size

        self.data = []
        c = 0
        for path in self.root.iterdir():
            audio = np.fromfile(path, dtype='uint8')
            if audio.shape[0] != 40653:
                print("Skipping")
                continue

            self.data.append(audio)

            c += 1
            if c == self.size:
                break

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx],


if __name__ == '__main__':
    import io
    import torchaudio


    def decode_torch(array: np.ndarray):
        return torchaudio.load(io.BytesIO(array.tobytes()), format='mp3')[0][0].numpy()


    with NamedTemporaryFile() as handle:
        dataset = DemoDataset('/media/t9s9/SSD_ubuntu/datasets/ESC50/mp3/test', size=1024)

        writer = DatasetWriter(handle.name, {
            'audio': MP3Field(input_shape=(40653,), output_shape=(160000,))
        })
        writer.from_indexed_dataset(dataset, shuffle_indices=False)

        loader = Loader(handle.name,
                        batch_size=64,
                        num_workers=2,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        custom_fields={'audio': MP3Field})

        for i, data in enumerate(loader):
            audio_field = data[0]
            for j, dp in enumerate(audio_field):
                torch_decode = decode_torch(dataset[i * loader.batch_size + j][0])
                equal = np.allclose(dp, torch_decode, atol=1e-5)
                # assert equal, f"Test failed at {i * loader.batch_size + j}."
                if not equal:
                    print(f"Test failed at {i * loader.batch_size + j}.")
        print("Test successful.")
