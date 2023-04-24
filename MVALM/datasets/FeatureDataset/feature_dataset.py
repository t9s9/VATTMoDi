import json
import random
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict, Callable

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FeatureOutput(NamedTuple):
    filename: str
    dataset: str
    caption: Union[List[str], str]
    image_embeddings: Union[torch.Tensor, np.ndarray]
    text_embeddings: Union[torch.Tensor, np.ndarray]
    audio_embeddings: Union[torch.Tensor, np.ndarray]
    image_projections: Optional[Union[torch.Tensor, np.ndarray]]
    text_projections: Optional[Union[torch.Tensor, np.ndarray]]
    spectrogram: Optional[Union[torch.Tensor, np.ndarray]] = None
    raw_audio: Optional[Union[torch.Tensor, np.ndarray]] = None
    image: Optional[Image.Image] = None


def build_extraction_dataset(root: str, splits: List[str], extraction_dir: str,
                             extra_annotations: Optional[str] = None, ):
    root = Path(root)
    for split in splits:
        base_annot = pd.read_parquet(root / f'annot_{split}.parquet')
        base_annot.filename = base_annot.filename.apply(lambda x: Path(x).stem)
        base_annot = base_annot.set_index('filename').to_dict(orient='index')

        for filetype_dir in (root / extraction_dir).iterdir():
            if filetype_dir.is_dir():
                for filename in (filetype_dir / split).iterdir():
                    if filename.stem in base_annot.keys():
                        base_annot[filename.stem][filetype_dir.stem] = str(filename.relative_to(root))
                    elif filename.stem[:25] in base_annot.keys():
                        if filetype_dir.stem not in base_annot[filename.stem[:25]].keys():
                            base_annot[filename.stem[:25]][filetype_dir.stem] = []
                        base_annot[filename.stem[:25]][filetype_dir.stem].append(str(filename.relative_to(root)))
                    else:
                        print(f'Could not find {filename.stem} in base_annot')
        base_annot = pd.DataFrame.from_dict(base_annot, orient='index')
        print(base_annot)
        quit()

        if extra_annotations is not None:
            extra_annot = pd.read_parquet(root / extra_annotations.format(split))
            extra_annot.filename = extra_annot.filename.apply(lambda x: Path(x).stem)
            extra_annot = extra_annot.set_index('filename')
            print(f"Extra annotations: {len(extra_annot)}, Base annotations: {len(base_annot)}")
            base_annot = base_annot.join(extra_annot, how="inner")
            print(f"New base annotations: {len(base_annot)}")

        print(base_annot)


class NewFeatureDataset(Dataset):
    def __init__(self, root: Union[str, Path],
                 split: str = 'train'):
        self.root = Path(root)
        self.split = split


class FeatureDataset(Dataset):
    """
    Dataset that contains extracted features from another dataset. This includes extracted images and audio clips as raw
    audio or spectrograms.
    """

    def __init__(self, root: Union[str, Path],
                 split: str = 'train',
                 return_spectrogram: bool = False,
                 return_raw_audio: bool = False,
                 return_image: bool = False,
                 return_projections: bool = False,
                 sample_caption: bool = False,
                 as_np: bool = False,
                 target_transform: Optional[Callable] = None, ):
        self.root = Path(root)
        self.split = split
        self.return_spectrogram = return_spectrogram
        self.return_raw_audio = return_raw_audio
        self.return_image = return_image
        self.return_projections = return_projections
        self.sample_caption = sample_caption
        self.as_np = as_np
        self.target_transform = target_transform

        self.name = None
        self.extraction_per_sample = None

        assert self.has_features, f'Features not found in {self.root}'
        assert return_spectrogram and self.has_spectrogram or not return_spectrogram, f'Spectrogram not found in {self.root}'
        assert (return_raw_audio and (
                self.has_raw or self.has_flac)) or not return_raw_audio, f'Raw audio not found in {self.root}'
        assert return_image and self.has_image or not return_image, f'Image not found in {self.root}'

        self.available_splits = list(map(lambda x: x.name, self.feature_path.iterdir()))
        assert self.split in self.available_splits, f'Split {self.split} not found in {self.available_splits}'

        self.features = list((self.feature_path / self.split).iterdir())
        self.check_features()
        random.seed(42)

    def check_features(self):
        feature = torch.load(self.features[0])
        self.name = feature['dataset']
        self.extraction_per_sample = feature['frame_id'].shape[0]

    def __getitem__(self, idx) -> FeatureOutput:
        spectrogram, raw_audio, image, image_projections, text_projections = None, None, None, None, None
        path = self.features[idx // self.extraction_per_sample]
        feature = torch.load(path)
        sample_idx = idx % self.extraction_per_sample

        image_embeddings = feature['image_embeddings'][sample_idx]
        audio_embeddings = feature['audio_embeddings'][sample_idx]
        text_embeddings = feature['text_embeddings']
        caption = feature['caption']

        if self.return_projections:
            image_projections = feature['image_features'][sample_idx]
            text_projections = feature['text_features']

        if self.sample_caption:
            random_text_idx = random.randint(0, text_embeddings.shape[0] - 1)
            text_embeddings = text_embeddings[random_text_idx]
            caption = caption[random_text_idx]

        if self.return_spectrogram and self.has_spectrogram:
            spectrogram = torch.load(self.spectrogram_path / self.split / path.name)[sample_idx]

        if self.return_raw_audio:
            filename = (path.stem + f"_{feature['top_sim_id'][sample_idx]}")
            if self.has_raw:
                raw_audio = torch.load((self.raw_path / self.split / filename).with_suffix('.pt'))[sample_idx]
            elif self.has_flac:
                raw_audio, sr = librosa.load((self.flac_path / self.split / filename).with_suffix('.flac'), sr=None,
                                             mono=True)

        if self.return_image and self.has_image:
            image = Image.open(
                self.image_path / self.split / (path.stem + f"_frame{feature['top_sim_id'][sample_idx]}" + '.png'))

        if self.as_np:
            image_embeddings = image_embeddings.numpy()
            audio_embeddings = audio_embeddings.numpy()
            text_embeddings = text_embeddings.numpy()
            if self.return_projections:
                image_projections = image_projections.numpy()
                text_projections = text_projections.numpy()
            if spectrogram is not None:
                spectrogram = spectrogram.numpy()
            if raw_audio is not None and self.has_raw:
                raw_audio = raw_audio.numpy()
            if image is not None and self.has_image:
                image = np.array(image)

        output = FeatureOutput(filename=feature['filename'], caption=caption, dataset=self.name,
                               image_embeddings=image_embeddings, text_embeddings=text_embeddings,
                               audio_embeddings=audio_embeddings, spectrogram=spectrogram,
                               raw_audio=raw_audio, image=image, image_projections=image_projections,
                               text_projections=text_projections)

        if self.target_transform is not None:
            output = self.target_transform(output)

        return output

    def __str__(self) -> str:
        return f'Feature dataset based on `{self.name}`, Split: `{self.split}`, Size: {len(self)}\nConfig: {self.config}'

    def __len__(self):
        return len(self.features) * self.extraction_per_sample

    @property
    @lru_cache
    def config(self) -> Dict:
        return json.load(open(self.root / 'config.json'))

    @property
    def feature_path(self) -> Path:
        return self.root / 'features'

    @property
    def has_features(self) -> bool:
        return self.feature_path.exists()

    @property
    def flac_path(self) -> Path:
        return self.root / 'flac_audio'

    @property
    def has_flac(self) -> bool:
        return self.flac_path.exists()

    @property
    def raw_path(self) -> Path:
        return self.root / 'raw_audio'

    @property
    def has_raw(self) -> bool:
        return self.raw_path.exists()

    @property
    def image_path(self) -> Path:
        return self.root / 'images'

    @property
    def has_image(self) -> bool:
        return self.image_path.exists()

    @property
    def spectrogram_path(self) -> Path:
        return self.root / 'spectrogram'

    @property
    def has_spectrogram(self) -> bool:
        return self.spectrogram_path.exists()


if __name__ == '__main__':
    build_extraction_dataset('/data/mmssl/AudioCaps/',
                             splits=['test'],
                             extraction_dir='extraction-10s',
                             extra_annotations='vis_captions/vis_caption_{}.parquet')
