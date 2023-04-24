from typing import Optional, Dict, List, Tuple, Union

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoModel, AutoTokenizer, BatchEncoding, ImageProcessingMixin

from MVALM.models import get_model_passt, MelSpectrogram, get_model_htsat


class Encoder(nn.Module):
    def __init__(self,
                 model_name: str,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.model_name = model_name
        self.gradient_checkpointing = gradient_checkpointing
        self.frozen = False
        self.hidden_size = None
        self.model: nn.Module = None

    def freeze(self):
        """ Freeze all params for inference. """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.frozen = True

    def unfreeze(self) -> None:
        """ Unfreeze all parameters for training."""
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.frozen = False

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __str__(self):
        return f"{self.model_name} - Parameters: {self.num_parameters:,};"f" Hidden Size: {self.hidden_size}"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        with torch.set_grad_enabled(not self.frozen):
            return self._forward(*args, **kwargs)


class VisionEncoder(Encoder):
    def __init__(self,
                 model_name: str,
                 gradient_checkpointing: bool = True,
                 cache_dir: Optional[str] = None,
                 model_class=None,
                 freeze: bool = False):
        super().__init__(model_name, gradient_checkpointing)
        self.source, self.model_name = model_name.split('-', 1)

        if self.source == 'timm':
            self.model = timm.create_model(self.model_name, pretrained=True)
            # remove linear layer at the end
            self.model.reset_classifier(num_classes=0)
            self.hidden_size = self.model.num_features

            if self.gradient_checkpointing:
                self.model.set_grad_checkpointing(True)
            # print("Vision encoder using gradient checkpointing:", self.model.grad_checkpointing)
        elif self.source == 'hf':

            model_kwargs = dict(cache_dir=cache_dir)

            model_cls = model_class if model_class is not None else AutoModel
            self.model = model_cls.from_pretrained(self.model_name, **model_kwargs)

            if self.gradient_checkpointing:
                print("Enabling gradient checkpointing for vision encoder")
                self.model.gradient_checkpointing_enable()

            self.hidden_size = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def preprocessor_kwargs(self) -> Dict:
        if self.source == 'timm':
            config = resolve_data_config({}, model=self.model)
            return {
                'input_size': config.pop('input_size', (3, 224, 224)),
                'mean': config.pop('mean', (0.5, 0.5, 0.5)),
                'std': config.pop('std', (0.5, 0.5, 0.5)),
                'crop_pct': config.pop('crop_pct', 0.9),
                'interpolation': config.pop('interpolation', 'bicubic'),
            }
        elif self.source == 'hf':
            config = ImageProcessingMixin.get_image_processor_dict(self.model_name)[0]
            return {
                'input_size': (3, config.pop('size', 224), config.pop('size', 224)),
                'mean': config.pop('image_mean', (0.48145466, 0.4578275, 0.40821073)),
                'std': config.pop('image_std', (0.26862954, 0.26130258, 0.27577711)),
                'crop_pct': config.pop('crop_size', 224) / config.pop('size', 224),
                'interpolation': config.pop('interpolation', 'bicubic'),
            }
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

    def get_preprocessor(self, is_training=False, **kwargs):
        transform_kwargs = self.preprocessor_kwargs()
        transform_kwargs.update(kwargs)
        return create_transform(**transform_kwargs, is_training=is_training)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == 'timm':
            return self.model(x)
        elif self.source == 'hf':
            return self.model(x).pooler_output


class TextEncoder(Encoder):
    def __init__(self,
                 model_name: str,
                 max_length: int = 60,
                 avg_word_embs: bool = True,
                 cache_dir: Optional[str] = None,
                 gradient_checkpointing: bool = True,
                 model_class=None,
                 tokenizer_class=None,
                 freeze: bool = False):
        super().__init__(model_name, gradient_checkpointing)
        self.max_length = max_length
        self.avg_word_embs = avg_word_embs

        model_kwargs = dict(cache_dir=cache_dir)

        model_cls = model_class if model_class is not None else AutoModel
        tokenizer_cls = tokenizer_class if tokenizer_class is not None else AutoTokenizer
        self.model = model_cls.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = tokenizer_cls.from_pretrained(model_name, use_fast=False)

        if self.gradient_checkpointing:
            print("Enabling gradient checkpointing for text encoder")
            self.model.gradient_checkpointing_enable()

        print("Text encoder using gradient checkpointing:", self.model.is_gradient_checkpointing)
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def tokenize(self, texts: Union[str, List[str], Tuple[str]], padding=True) -> BatchEncoding:
        if type(texts) == tuple:
            texts = list(texts)

        tokens = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_attention_mask=True
        )

        return tokens

    def _forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor:
        sequence_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.avg_word_embs:
            attention_mask = attention_mask.unsqueeze(-1)
            return torch.sum(sequence_output.last_hidden_state * attention_mask, dim=1) / (
                torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9))
        else:
            return sequence_output.pooler_output


class AudioEncoder(Encoder):
    def __init__(self,
                 model_name: str = 'passt_s_swa_p16_128_ap476',
                 gradient_checkpointing: bool = True,
                 spectrogram_parameters: Dict = None,
                 cache_dir: str = None,
                 freeze: bool = False,
                 **model_kwargs):
        super().__init__(model_name, gradient_checkpointing)
        self.gradient_checkpointing = model_kwargs.pop('gradient_checkpointing', gradient_checkpointing)

        if self.model_name.startswith('passt'):
            spectrogram_defaults = dict(n_fft=1024, win_length=800, hopsize=320, sr=32000, n_mels=128, fmin=0,
                                        fmax=None)
            if spectrogram_parameters is not None:
                spectrogram_defaults.update(spectrogram_parameters)

            self.spectrogram = MelSpectrogram(**spectrogram_defaults)
            self.model = get_model_passt(arch=self.model_name,
                                         gradient_checkpointing=self.gradient_checkpointing,
                                         **model_kwargs)
            self.hidden_size = self.model.embed_dim
        elif self.model_name.startswith('htsat'):
            self.model = get_model_htsat(arch=self.model_name,
                                         gradient_checkpointing=self.gradient_checkpointing,
                                         cache_dir=cache_dir,
                                         num_classes=0,
                                         **model_kwargs)

            self.spectrogram = self.model.forward_spectrogram
            self.hidden_size = self.model.num_features
        else:
            raise ValueError(f"Unknown audio model {self.model_name}.")

        if freeze:
            self.freeze()
            self.model.train(True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the audio embedding for given audio tensor.

        Args:
            x: For PASST: shape [batch_size, 1, freq_bins, time_bins]
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        return self.model.forward_feature(x)
