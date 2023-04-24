from pathlib import Path

import torch
from ffcv.fields import IntField, BytesField, NDArrayField
from ffcv.writer import DatasetWriter
from MVALM.datasets.AudioSet.audioset import AudioSetAudioOnly
import numpy as np
from MVALM.datasets import UrbanSound8k, ESC50, VGGSoundAudio, BDLib2, FSD50k, TAU
from MVALM.datasets.utils import multi_one_hot
from MVALM.models.PASST import AugmentMelSTFT, MelSpectrogram
from MVALM.models.encoder import AudioEncoder
from MVALM.datasets.base import AudioDataset

if __name__ == '__main__':
    model = AudioEncoder(model_name='passt_s_swa_p16_128_ap476', gradient_checkpointing=False).eval()
    spec = model.spectrogram


    class Wrapper:
        def __init__(self, dataset: AudioDataset, transform=None, multi_label=False):
            self.dataset = dataset
            self.transform = transform
            self.multi_label = multi_label

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            dp = self.dataset[idx]
            audio = dp.audio
            label = dp.label

            if self.transform is not None:
                with torch.no_grad():
                    audio = self.transform(audio.unsqueeze(0)).squeeze()

            if self.multi_label:
                label = multi_one_hot(label, self.dataset.num_classes)

            return audio.numpy(), label


    # for split in ('train', 'val', 'test'):
    #     dataset = FSD50k(datasets_root='/media/t9s9/SSD_ubuntu/datasets/', split=split).add_source('spectrogram')
    #     multi_label = True
    #     fields = {
    #         'spectrogram': NDArrayField(shape=(128, 1000), dtype=np.dtype('float32')),
    #         'label': NDArrayField(shape=(dataset.num_classes,), dtype=np.dtype('int'))
    #     }
    #     output_path = Path('/home/t9s9/Datasets/beton/evaluation') / (dataset.name + f'{split}-spec.beton')
    #     writer = DatasetWriter(str(output_path), fields, num_workers=8)
    #     writer.from_indexed_dataset(Wrapper(dataset, None, multi_label, 'spectrogram'), shuffle_indices=False)

    # dataset = VGGSoundAudio(datasets_root='/home/t9s9/Datasets/', sample_rate=32000, mono=True, length=320000)
    # multi_label = False
    # fields = {
    #     'audio': NDArrayField(shape=(128, 1000), dtype=np.dtype('float32')),
    #     'label': IntField()
    # }

    # dataset = AudioSetAudioOnly(split='val', verbose=True, sample_rate=32000, length=32000 * 10, mono=True)
    # dataset.verify_files()
    # multi_label = True
    # fields = {
    #     'audio': NDArrayField(shape=(128, 1000), dtype=np.dtype('float32')),
    #     'label': NDArrayField(shape=(dataset.num_classes,), dtype=np.dtype('int'), )
    # }

    #
    # dataset = ESC50(datasets_root='/media/t9s9/SSD_ubuntu/datasets', sample_rate=32000, mono=True)
    # multi_label = False
    # spec_shape = (501, 64) if model.model_name == 'htsat' else (128, 500)

    # dataset = TAU(datasets_root='/media/t9s9/SSD_ubuntu/datasets', sample_rate=32000, mono=True, length=320000)
    # multi_label = False
    # spec_shape = (128, 1000)

    # dataset = UrbanSound8k(datasets_root='/media/t9s9/SSD_ubuntu/datasets', sample_rate=32000, mono=True, length=128000)
    # multi_label = False
    # spec_shape(128, 400)

    # dataset = BDLib2(datasets_root='/home/t9s9/Datasets/', sample_rate=32000, mono=True, length=320000)
    # multi_label = False
    # fields = {
    #     'audio': NDArrayField(shape=(128, 1000), dtype=np.dtype('float32')),
    #     'label': IntField()
    # }

    fields = {
        'audio': NDArrayField(shape=spec_shape, dtype=np.dtype('float32')),
        'label': IntField()
    }

    output_path = Path('/home/t9s9/Datasets/beton/evaluation') / (dataset.name + f'-{model.model_name}-spec.beton')
    writer = DatasetWriter(str(output_path), fields, num_workers=2)
    writer.from_indexed_dataset(Wrapper(dataset, spec, multi_label), shuffle_indices=False)
