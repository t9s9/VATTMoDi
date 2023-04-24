import random
from pathlib import Path
from typing import Tuple, List, Union, Type, Dict, Any, NamedTuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import GPT2Tokenizer, T5Tokenizer, BatchEncoding, AutoTokenizer, RobertaTokenizer

from MVALM.datasets.base import AudioCaptionDataset
from helper import tokenize, pad


class AudioCaptionOutput(NamedTuple):
    audio_embedding: torch.Tensor
    spectrogram: Optional[torch.Tensor]
    filename: str
    caption: Union[str, List[str]]  # depends on choose_caption
    tags: List[str]


class AudioCapDatasetWrapper(Dataset):
    def __init__(self,
                 dataset: AudioCaptionDataset,
                 choose_caption: bool = True,
                 name: str = None,
                 load_spectrogram: bool = False):
        self.dataset = dataset
        self.embedding_path = dataset.root / 'embeddings'
        self.spectrogram_path = dataset.root / 'spectrogram'
        self.load_spectrogram = load_spectrogram

        self.choose_caption = choose_caption
        self.name = dataset.name if name is None else name
        random.seed(42)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> AudioCaptionOutput:
        """
        Returns: Audio embedding, caption, tags
        """
        # use df to not load audio
        dp = self.dataset.annot.iloc[i]

        caption = dp[self.dataset.LABEL_COLUMN]
        filename = str(dp['filename'])
        if isinstance(caption, (list, np.ndarray)) and self.choose_caption:
            caption = random.choice(caption)

        embedding = torch.load(self.embedding_path / Path(filename).with_suffix('.pt'))
        spectrogram = torch.load(
            self.spectrogram_path / Path(filename).with_suffix('.pt')) if self.load_spectrogram else None

        return AudioCaptionOutput(audio_embedding=embedding, spectrogram=spectrogram, filename=filename,
                                  caption=caption, tags=dp.tags)


class PostProcessor(Dataset):
    def __init__(self, dataset: Union[AudioCaptionDataset, ConcatDataset]):
        super().__init__()
        self.dataset = dataset

    def outputs(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        return NotImplementedError()

    def verify(self):
        print('Verifying dataset...')
        sample = self[0]
        fields = list(self.outputs().keys())
        gt = list(self.outputs().values())
        for i, obj in enumerate(sample):
            assert obj.shape == gt[i][0], f'Expected shape {gt[i][0]}, got {obj.shape} for field {fields[i]}'
            assert obj.dtype == gt[i][1], f'Expected shape {gt[i][1]}, got {obj.dtype} for field {fields[i]}'

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i) -> Any:
        return NotImplementedError()


class AudioCaptionPostprocessor(PostProcessor):
    def __init__(self,
                 dataset: Union[AudioCaptionDataset, ConcatDataset],
                 token_max_length: int = 77,
                 caption_model: str = 'gpt2',
                 **kwargs):
        super().__init__(dataset)
        self.tokenizer = GPT2Tokenizer.from_pretrained(caption_model)
        self.token_max_length = token_max_length

        self.verify()

    def outputs(self):
        return {'audio_embeddings': ((768,), np.dtype('float32')),
                'tokens': ((self.token_max_length,), np.dtype('int64')),
                'mask': ((self.token_max_length,), np.dtype('float32'))}

    def __getitem__(self, i) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dp: AudioCaptionOutput = self.dataset[i]

        tokens = tokenize(dp.caption, self.tokenizer)
        tokens, mask = pad(tokens, self.token_max_length)

        return dp.audio_embedding.numpy(), tokens.numpy(), mask.numpy()


class AudioSpectrogramCaptionPostprocessor(PostProcessor):
    def __init__(self,
                 dataset: Union[AudioCaptionDataset, ConcatDataset],
                 token_max_length: int = 77,
                 spectrogram_shape: Tuple[int, int] = (128, 1000),
                 caption_model: str = 'gpt2',
                 **kwargs):
        super().__init__(dataset)
        self.tokenizer = GPT2Tokenizer.from_pretrained(caption_model)
        self.token_max_length = token_max_length
        self.s_h, self.s_w = spectrogram_shape

        self.verify()

    def outputs(self):
        return {'spectrogram': ((self.s_h, self.s_w), np.dtype('float32')),
                'tokens': ((self.token_max_length,), np.dtype('int64')),
                'mask': ((self.token_max_length,), np.dtype('float32'))}

    def __getitem__(self, i) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dp: AudioCaptionOutput = self.dataset[i]

        tokens = tokenize(dp.caption, self.tokenizer)
        tokens, mask = pad(tokens, self.token_max_length)

        spectrogram = dp.spectrogram.numpy()
        if spectrogram.shape[1] < self.s_w:
            zeros = np.zeros(shape=(self.s_h, self.s_w - spectrogram.shape[1]))
            spectrogram = np.concatenate([spectrogram, zeros], axis=1)
        else:
            spectrogram = spectrogram[:, :self.s_w]

        return spectrogram, tokens.numpy(), mask.numpy()


class AudioTagCaptionPostprocessor(PostProcessor):
    def __init__(self,
                 dataset: Union[AudioCaptionDataset, ConcatDataset],
                 num_tag_variants: int = 1,
                 token_max_length: int = 77,
                 caption_model: str = 'gpt2',
                 tag_token_max_length: int = 60,
                 tag_model: str = 't5-small',
                 **kwargs):
        super().__init__(dataset)
        self.tokenizer = GPT2Tokenizer.from_pretrained(caption_model)
        if tag_model == 't5-base':
            self.tag_tokenizer = T5Tokenizer.from_pretrained(tag_model)
        elif tag_model == 'roberta-base':
            self.tag_tokenizer = RobertaTokenizer.from_pretrained(tag_model)
        else:
            raise ValueError(f'Unknown model name: {tag_model}')
        self.token_max_length = token_max_length
        self.tag_token_max_length = tag_token_max_length
        self.num_tag_variants = num_tag_variants

        self.verify()

    def outputs(self):
        return {
            'audio_embeddings': ((768,), np.dtype('float32')),
            'tag_tokens': ((self.num_tag_variants, self.tag_token_max_length), np.dtype('int64')),
            'tag_mask': ((self.num_tag_variants, self.tag_token_max_length), np.dtype('int64')),
            'tokens': ((self.token_max_length,), np.dtype('int64')),
            'mask': ((self.token_max_length,), np.dtype('float32')),
        }

    def _tokenize_tags(self, tags: Union[List[str], List[List[str]]]):
        clean = lambda x: x.replace('_', ' ').replace(',', '').strip() if isinstance(x, str) else list(map(clean, x))
        tags = list(map(clean, tags))
        if isinstance(tags[0], str):
            tags = [tags]
        tags = list(map(lambda x: ' '.join(x), tags))
        return self.tag_tokenizer(tags,
                                  max_length=self.tag_token_max_length,
                                  return_tensors="pt",
                                  padding='max_length',
                                  truncation=True)

    def __getitem__(self, i) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dp: AudioCaptionOutput = self.dataset[i]

        tokens = tokenize(dp.caption, self.tokenizer)
        tokens, mask = pad(tokens, self.token_max_length)
        if self.num_tag_variants > 1:
            tag_tokens = self._tokenize_tags(
                [list(np.random.permutation(dp.tags)) for _ in range(self.num_tag_variants)])
        else:
            tag_tokens = self._tokenize_tags(dp.tags)

        return (dp.audio_embedding.numpy(), tag_tokens.input_ids.numpy(), tag_tokens.attention_mask.numpy(),
                tokens.numpy(), mask.numpy())


def create_audio_beton(train_ds: Dataset, val_ds: Dataset, postprocessor: Type[PostProcessor], output_path: str,
                       postprocessor_kwargs: Dict = {}, num_workers: int = 16, shuffle_indices: bool = False) -> None:
    from ffcv.fields import NDArrayField
    from ffcv.writer import DatasetWriter
    random.seed(42)

    if train_ds is not None:
        train_ds = postprocessor(train_ds, **postprocessor_kwargs)
        train_fields = {name: NDArrayField(dtype=dtype, shape=shape)
                        for name, (shape, dtype) in train_ds.outputs().items()}

        writer = DatasetWriter(str(output_path) + '_train.beton', train_fields, num_workers=num_workers)
        writer.from_indexed_dataset(train_ds, shuffle_indices=shuffle_indices)

    if val_ds is not None:
        val_ds = postprocessor(val_ds, **postprocessor_kwargs)
        val_fields = {name: NDArrayField(dtype=dtype, shape=shape)
                      for name, (shape, dtype) in val_ds.outputs().items()}

        writer = DatasetWriter(str(output_path) + '_val.beton', val_fields, num_workers=num_workers)
        writer.from_indexed_dataset(val_ds, shuffle_indices=False)


def create_datasets(wrapper, datasets: List[Tuple[str, Type[AudioCaptionDataset]]],
                    wrapper_kwargs: Dict = {}) -> Tuple[Dataset, Dataset]:
    train_ds, val_ds = [], []
    for root, dataset in datasets:
        print(f'Loading {dataset.__name__}...')
        splits = dataset.LABELS['caption'].keys()
        for split in splits:
            if split == 'val':
                val_ds.append(wrapper(dataset(datasets_root=root, split='val', expand_captions=False),
                                      **wrapper_kwargs))
            elif split == 'train':
                train_ds.append(wrapper(dataset(datasets_root=root, split='train', expand_captions=True),
                                        **wrapper_kwargs))
    print(f'Loaded {len(train_ds)} train datasets and {len(val_ds)} val datasets')
    return ConcatDataset(train_ds) if train_ds else None, ConcatDataset(val_ds) if val_ds else None
