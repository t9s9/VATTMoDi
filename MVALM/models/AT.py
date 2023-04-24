from typing import Optional, NamedTuple, Tuple

import torch
import torch.nn as nn

from .backbone.layers import Mlp
from .encoder import AudioEncoder, TextEncoder


class ATProj(NamedTuple):
    a_proj: torch.Tensor
    t_proj: torch.Tensor


class ATProjectionHead(nn.Module):
    def __init__(self,
                 hidden_sizes: Tuple[int, int],
                 proj_dim: int,
                 bias: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        self.text_proj = Mlp(in_features=hidden_sizes[0], hidden_features=2 * proj_dim, out_features=proj_dim,
                             bias=bias, dropout=dropout)
        self.audio_proj = Mlp(in_features=hidden_sizes[1], hidden_features=2 * proj_dim, out_features=proj_dim,
                              bias=bias, dropout=dropout)

    def forward(self,
                text_embed: Optional[torch.Tensor] = None,
                audio_embed: Optional[torch.Tensor] = None):
        t, a = None, None
        if text_embed is not None:
            t = self.text_proj(text_embed)
        if audio_embed is not None:
            a = self.audio_proj(audio_embed)
        return ATProj(a_proj=a, t_proj=t)


class AT(nn.Module):
    def __init__(self,
                 text_encoder: TextEncoder,
                 audio_encoder: AudioEncoder,
                 proj_dim: int = 512,
                 ):
        super().__init__()
        self.proj_dim = proj_dim
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

        self.head = ATProjectionHead(hidden_sizes=(audio_encoder.hidden_size,
                                                   text_encoder.hidden_size),
                                     proj_dim=proj_dim)

    @classmethod
    def from_config(cls, config):
        text_encoder = TextEncoder(**config['text_encoder'])
        audio_encoder = AudioEncoder(**config['audio_encoder'])

        return cls(text_encoder, audio_encoder, **config['at'])

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def encode_audio(self, audio: torch.Tensor = None, spectrogram=None) -> torch.Tensor:
        if audio is not None:
            with torch.cuda.amp.autocast(enabled=False):
                spectrogram = self.audio_encoder.spectrogram(audio)
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.unsqueeze(1)
        return self.audio_encoder(spectrogram)

    def forward(self, input_ids=None, attention_mask=None, audio=None, spectrogram=None) -> ATProj:
        audio = self.encode_audio(audio, spectrogram) if (
                audio is not None or spectrogram is not None) else None

        text = self.encode_text(input_ids, attention_mask) if (
                input_ids is not None and attention_mask is not None) else None

        return self.head(text_embed=text, audio_embed=audio)

    def __str__(self):
        return f"--- VAT ---\n" \
               f"\nText Encoder: {self.text_encoder}" \
               f"\nAudio Encoder: {self.audio_encoder}"
