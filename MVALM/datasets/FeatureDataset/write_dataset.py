import collections
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import List, Type, Tuple, Union

import numpy as np
from ffcv.fields import NDArrayField, RGBImageField
from ffcv.writer import DatasetWriter
from torch.utils.data import ConcatDataset

from MVALM.datasets.FeatureDataset.feature_dataset import FeatureOutput
from MVALM.datasets.base import VideoCaptionDataset
from MVALM.datasets.custom_ffcv_fields import FlexNDArrayField
from MVALM.models import CLIPTokenizer


class ImageInputType(Enum):
    IMAGE = 'image'
    EMBEDDINGS = 'image_embeddings'
    PROJECTIONS = 'image_projections'


class TextInputType(Enum):
    TEXT = 'text'
    TOKENS = 'tokens'
    EMBEDDINGS = 'text_embeddings'
    PROJECTIONS = 'text_projections'


class AudioInputType(Enum):
    RAW = 'raw'
    EMBEDDINGS = 'audio_embeddings'
    SPECTROGRAM = 'spectrogram'


class ImageConfig:
    def __init__(self, image_type: ImageInputType):
        self.image_type = image_type

    def prepare(self, dp: FeatureOutput) -> np.ndarray:
        if self.image_type == self.image_type.EMBEDDINGS:
            return dp.image_embeddings.astype('float32')
        elif self.image_type == self.image_type.PROJECTIONS:
            return dp.image_projections.astype('float32')
        elif self.image_type == self.image_type.IMAGE:
            return dp.image

    def get_field(self):
        if self.image_type == self.image_type.EMBEDDINGS:
            return NDArrayField(shape=(768,), dtype=np.dtype('float32'))
        elif self.image_type == self.image_type.PROJECTIONS:
            return NDArrayField(shape=(512,), dtype=np.dtype('float32'))
        elif self.image_type == self.image_type.IMAGE:
            return RGBImageField(write_mode='raw')


class TextConfig:
    def __init__(self, text_type: TextInputType, max_token_length: int = 77, sample_caption: bool = False):
        self.text_type = text_type
        self.max_token_length = max_token_length
        self.sample_caption = sample_caption
        self.tokenizer = CLIPTokenizer(context_length=self.max_token_length, truncate=True)

    def prepare(self, dp: FeatureOutput) -> np.ndarray:
        if self.text_type == self.text_type.EMBEDDINGS:
            return dp.text_embeddings.astype('float32')
        elif self.text_type == self.text_type.PROJECTIONS:
            return dp.text_projections.astype('float32')
        elif self.text_type == self.text_type.TEXT:
            return self.tokenizer(dp.caption).numpy().astype('int32')

    def get_field(self):
        if self.text_type == self.text_type.EMBEDDINGS:
            if self.sample_caption:
                return NDArrayField(shape=(768,), dtype=np.dtype('float32'))
            else:
                return FlexNDArrayField(fixed_dim=768, dtype=np.dtype('float32'))
        elif self.text_type == self.text_type.PROJECTIONS:
            if self.sample_caption:
                return NDArrayField(shape=(768,), dtype=np.dtype('float32'))
            else:
                return FlexNDArrayField(fixed_dim=768, dtype=np.dtype('float32'))
        elif self.text_type == self.text_type.TEXT:
            if self.sample_caption:
                return NDArrayField(shape=(self.max_token_length,), dtype=np.dtype('int32'))
            else:
                return FlexNDArrayField(fixed_dim=self.max_token_length, dtype=np.dtype('int32'))


class AudioConfig:
    def __init__(self, audio_type: AudioInputType, spectrogram_shape: tuple = (128, 400)):
        self.audio_type = audio_type
        self.spec_width = spectrogram_shape[1]
        self.spec_height = spectrogram_shape[0]
        print(f'AudioConfig: {self.audio_type=}, {self.spec_width=}, {self.spec_height=}')

    def prepare(self, dp: FeatureOutput) -> np.ndarray:
        if self.audio_type == self.audio_type.EMBEDDINGS:
            return dp.audio_embeddings.astype('float32')
        elif self.audio_type == self.audio_type.SPECTROGRAM:
            spectrogram = dp.spectrogram
            if spectrogram.shape[1] < self.spec_width:
                zeros = np.zeros(shape=(self.spec_height, self.spec_width - spectrogram.shape[1]))
                spectrogram = np.concatenate([spectrogram, zeros], axis=1)
            else:
                spectrogram = spectrogram[:, :self.spec_width]
            return spectrogram.astype('float32')
        elif self.audio_type == self.audio_type.RAW:
            # TODO: pad & truncate
            return dp.raw_audio.astype('float32')

    def get_field(self):
        if self.audio_type == self.audio_type.EMBEDDINGS:
            return NDArrayField(shape=(768,), dtype=np.dtype('float32'))
        elif self.audio_type == self.audio_type.SPECTROGRAM:
            return NDArrayField(shape=(self.spec_height, self.spec_width), dtype=np.dtype('float32'))
        elif self.audio_type == self.audio_type.RAW:
            return NDArrayField(shape=(1, 16000), dtype=np.dtype('float32'))


def create_ffcv_ds(datasets: List[Type[VideoCaptionDataset]],
                   dataset_root: str,
                   output_path: str = '/data/mmssl/beton/',
                   splits: Tuple[str, ...] = ('test', 'val', 'train'),
                   name: str = 'vat-4s-1i-embed',
                   sample_caption: Union[Tuple[bool, ...], bool] = False,
                   audio_type: AudioInputType = AudioInputType.EMBEDDINGS,
                   text_type: TextInputType = TextInputType.EMBEDDINGS,
                   image_type: ImageInputType = ImageInputType.EMBEDDINGS):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    if not isinstance(sample_caption, collections.abc.Iterable):
        sample_caption = tuple(repeat(sample_caption, 2))

    for i, split in enumerate(splits):
        print(sample_caption[i])
        image_conf = ImageConfig(image_type)
        text_conf = TextConfig(text_type, sample_caption=sample_caption[i])
        audio_conf = AudioConfig(audio_type)

        def get_fields(x):
            out = image_conf.prepare(x), text_conf.prepare(x), audio_conf.prepare(x)
            return out

        fields = {'image': image_conf.get_field(), 'text': text_conf.get_field(), 'audio': audio_conf.get_field()}

        ds_split = [ds.as_feature_dataset(datasets_root=dataset_root,
                                          feature_dir='extraction-4s',
                                          split=split,
                                          return_spectrogram=audio_type == AudioInputType.SPECTROGRAM,
                                          return_image=image_type == ImageInputType.IMAGE,
                                          return_raw_audio=audio_type == AudioInputType.RAW,
                                          return_projections=ImageInputType.PROJECTIONS == image_type or TextInputType.PROJECTIONS == text_type,
                                          as_np=True,
                                          sample_caption=sample_caption[i],
                                          target_transform=get_fields) for ds in datasets
                    if split in ds.LABELS['video'].keys()]
        dataset = ConcatDataset(ds_split)
        writer = DatasetWriter(str(output_path / f'{name}_{split}.beton'), fields, num_workers=2)
        writer.from_indexed_dataset(dataset, shuffle_indices=True)
