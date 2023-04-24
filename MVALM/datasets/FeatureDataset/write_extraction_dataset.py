from typing import List, Dict

import numpy as np
from ffcv.fields import NDArrayField, RGBImageField
from ffcv.writer import DatasetWriter
from torch.utils.data import ConcatDataset

from MVALM.datasets.FeatureDataset.extraction_dataset import ExtractionDataset, ExtractionOutput
from MVALM.datasets.custom_ffcv_fields import FlexNDArrayField
from MVALM.datasets.utils import pad_or_truncate
from MVALM.models import CLIPTokenizer


class FFCVPreparer:
    def __init__(self,
                 token_length: int = 77,
                 spectrogram_width: int = 1000,
                 train: bool = True):
        self.token_length = token_length
        self.spectrogram_width = spectrogram_width
        self.train = train

        self.tokenizer = CLIPTokenizer(context_length=self.token_length, truncate=True)

    def __call__(self, out: ExtractionOutput):
        images = out.images
        visual_caption = self.tokenizer(out.visual_captions).numpy().astype('int32')
        auditory_captions = self.tokenizer(out.auditory_captions).numpy().astype('int32')
        spectrogram = out.spectrogram
        if spectrogram.shape[1] < self.spectrogram_width:
            zeros = np.zeros(shape=(128, self.spectrogram_width - spectrogram.shape[1]))
            spectrogram = np.concatenate([spectrogram, zeros], axis=1).astype('float32')
        else:
            spectrogram = spectrogram[:, :self.spectrogram_width].astype('float32')

        if self.train:
            return visual_caption, auditory_captions, spectrogram, *images
        else:
            # take first captions and middle image
            return visual_caption[0], auditory_captions[0], spectrogram, images[len(images) // 2]


def ffcv_field(nr_images=3, token_length=77, spectrogram_width=1000, train=True) -> Dict:
    out = {}
    if train:
        out['visual_caption'] = FlexNDArrayField(fixed_dim=token_length, dtype=np.dtype('int32'))
        out['auditory_caption'] = FlexNDArrayField(fixed_dim=token_length, dtype=np.dtype('int32'))
        out['spectrogram'] = NDArrayField(shape=(128, spectrogram_width), dtype=np.dtype('float32'))
        for i in range(nr_images):
            out[f'image_{i}'] = RGBImageField(write_mode='jpg')
    else:
        out['visual_caption'] = NDArrayField(shape=(token_length,), dtype=np.dtype('int32'))
        out['auditory_caption'] = NDArrayField(shape=(token_length,), dtype=np.dtype('int32'))
        out['spectrogram'] = NDArrayField(shape=(128, spectrogram_width), dtype=np.dtype('float32'))
        out['image_0'] = RGBImageField(write_mode='jpg')
    return out


def write_extraction_dataset(datasets: List[ExtractionDataset],
                             output_path: str,
                             split: str,
                             nr_images: int = 3,
                             token_length: int = 77,
                             spectrogram_width: int = 1000):
    preparer = FFCVPreparer(token_length=token_length, spectrogram_width=spectrogram_width, train=split == 'train')
    for dataset in datasets:
        dataset.transform = preparer

    dataset = ConcatDataset(datasets)
    writer = DatasetWriter(output_path,
                           fields=ffcv_field(nr_images, token_length, spectrogram_width, train=split == 'train'),
                           num_workers=16)
    writer.from_indexed_dataset(dataset, shuffle_indices=True)
