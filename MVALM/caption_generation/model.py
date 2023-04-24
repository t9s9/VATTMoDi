from enum import Enum
from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, T5EncoderModel, T5Tokenizer, RobertaModel, RobertaTokenizer, OPTForCausalLM

from MVALM.models.VAT import TextEncoder, AudioEncoder
from MVALM.models.backbone.mem_eff_transformer import Transformer
from dataset import AudioCaptionOutput


class PrexfixMappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'
    CrossTransformer = 'cross-transformer'


class MLP(nn.Sequential):
    def __init__(self, sizes: Tuple[int, ...], bias: bool = True):
        super().__init__()

        for i in range(len(sizes) - 1):
            self.add_module(f'linear_{i}', nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                self.add_module(f'act_{i}', nn.Tanh())

        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if bias:
                    nn.init.zeros_(layer.bias)


class TransformerMapper(nn.Module):
    def __init__(self,
                 input_dim: Tuple[int, ...],
                 const_prefix_length: int,
                 embedding_prefix_length: Tuple[int, ...],
                 dim_embedding: int,
                 num_layers: int = 8,
                 num_heads: int = 8):
        super().__init__()
        assert len(input_dim) == len(embedding_prefix_length)
        self.total_embedding_prefix_length = sum(embedding_prefix_length)

        self.transformer = Transformer(dim_embedding, num_heads, num_layers, mlp_hidden_layer_multiplier=2,
                                       proj_drop=0.1)
        self.proj = nn.Linear(sum(input_dim), self.total_embedding_prefix_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(const_prefix_length, dim_embedding), requires_grad=True)

        # self.token_type_embeddings = nn.Embedding(2, dim_embedding)
        # self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        # get constant embedding for every sample in batch
        const_prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        # project input embeddings embedding
        x = self.proj(x).view(x.shape[0], self.total_embedding_prefix_length, -1)
        # merge constant and projected embeddings
        # shape: (total_embedding_prefix_length + const_prefix_length, dim_embedding)
        prefix = torch.cat((x, const_prefix), dim=1)

        out = self.transformer(prefix)[:, self.total_embedding_prefix_length:]
        return out


class CaptioningModelHead(nn.Module):
    def __init__(self,
                 input_dim: Tuple[int, ...],
                 const_prefix_length: int,
                 prefix_mapping_type: PrexfixMappingType = PrexfixMappingType.MLP,
                 embedding_prefix_length: Tuple[int, ...] = None,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 caption_model: str = 'gpt2'
                 ):
        super().__init__()
        self.const_prefix_length = const_prefix_length

        if caption_model.startswith('gpt2'):
            self.caption_model = GPT2LMHeadModel.from_pretrained(caption_model)
            self.wte = self.caption_model.transformer.wte
            self.wte_dim = self.wte.embedding_dim  # .weight.shape[1]
        elif caption_model.startswith('facebook/opt'):
            self.caption_model = OPTForCausalLM.from_pretrained(caption_model)
            self.wte = self.caption_model.get_input_embeddings()
            self.wte_dim = self.wte.embedding_dim
        else:
            raise ValueError(f'Caption model {caption_model} not supported.')

        if prefix_mapping_type == PrexfixMappingType.MLP:
            self.prefix_proj = MLP(
                (sum(input_dim),
                 (self.wte_dim * const_prefix_length) // 2,
                 self.wte_dim * const_prefix_length)
            )
        elif prefix_mapping_type == PrexfixMappingType.Transformer:
            self.prefix_proj = TransformerMapper(
                input_dim=input_dim,
                const_prefix_length=const_prefix_length,
                dim_embedding=self.wte_dim,
                embedding_prefix_length=embedding_prefix_length,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        elif prefix_mapping_type == PrexfixMappingType.CrossTransformer:
            raise NotImplementedError()

        print(f"CaptioningModelHead: {type(self.caption_model).__name__}({self.wte_dim=}), "
              f"PrefixProjection: {prefix_mapping_type.name}")

    def project_prefix(self, embeddings: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor) and embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        if isinstance(embeddings, (list, tuple)):
            if embeddings[0].ndim == 1:
                embeddings = [e.unsqueeze(0) for e in embeddings]
            embeddings = torch.cat(embeddings, dim=1)

        return self.prefix_proj(
            embeddings.type(self.wte.weight.dtype)
        ).view(-1, self.const_prefix_length, self.wte_dim)

    def forward(self,
                embeddings: Union[List[torch.Tensor], Tuple[torch.Tensor]],  # (batch_size, embedding_size)
                tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ):
        prefix_projections = (
            self.prefix_proj(
                torch.cat(embeddings, dim=1)
            )  # (batch_size, const_prefix_length * wte_dim)
            .view(-1, self.const_prefix_length, self.wte_dim)
            # (batch_size, const_prefix_length, wte_dim)
        )

        embedding_text = self.wte(tokens)  # word text embedding
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        masks = torch.cat(
            (
                torch.ones(mask.shape[0], self.const_prefix_length, device=prefix_projections.device),
                mask
            ), dim=1)  # adding prefix mask

        out = self.caption_model(inputs_embeds=embedding_cat, attention_mask=masks)
        return out


class Encoder(nn.Module):
    name: str = 'base-encoder'

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def infer(self, inputs: AudioCaptionOutput, device) -> Tuple[torch.Tensor]:
        return NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SingleEmbeddingEncoder(Encoder):
    name: str = 'embedding'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def infer(self, inputs: AudioCaptionOutput, device) -> Tuple[torch.Tensor]:
        return inputs.audio_embedding.to(device),

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor]:
        return embedding,


class TagEncoder(Encoder):
    name: str = 'tag'

    def __init__(self, model_name: str = 'roberta-base', **kwargs):
        super().__init__(**kwargs)
        if model_name == 't5-base':
            model_class, tokenizer_class = T5EncoderModel, T5Tokenizer
        elif model_name == 'roberta-base':
            model_class, tokenizer_class = RobertaModel, RobertaTokenizer
        else:
            raise ValueError(f'Unknown model name: {model_name}')

        self.encoder = TextEncoder(model_class=model_class,
                                   tokenizer_class=tokenizer_class,
                                   avg_word_embs=True,
                                   max_length=60,
                                   model_name=model_name)
        print("Tag encoder uses:", self.encoder)

    def infer(self, inputs: AudioCaptionOutput, device) -> torch.Tensor:
        tags = list(map(lambda x: x.replace('_', ' ').replace(',', '').strip(), inputs.tags))
        inputs = self.encoder.tokenizer([' '.join(tags)],
                                        max_length=self.encoder.max_length,
                                        return_tensors="pt",
                                        padding='max_length',
                                        truncation=True).to(device)

        return self(inputs.input_ids, inputs.attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 3 and attention_mask.ndim == 3:
            # sample one tag from each batch
            idx = torch.randint(input_ids.shape[1], size=(1,), dtype=torch.long).item()
            input_ids = input_ids[:, idx, :]
            attention_mask = attention_mask[:, idx, :]

        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


class SpectrogramEncoder(Encoder):
    name = 'spectrogram'

    def __init__(self,
                 s_patchout_t: int = 0,
                 s_patchout_f: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.s_patchout_f, self.s_patchout_t = s_patchout_f, s_patchout_t

        self.encoder = AudioEncoder(model_name='passt_s_swa_p16_128_ap476', s_patchout_t=s_patchout_t,
                                    s_patchout_f=s_patchout_f)
        del self.encoder.spectrogram

    def freeze(self):
        self.encoder.freeze()
        # train mode to enable patchout
        if self.s_patchout_f > 0 or self.s_patchout_t > 0:
            self.encoder.train()

    def infer(self, inputs: AudioCaptionOutput, device) -> Tuple[torch.Tensor]:
        return self(inputs.spectrogram.to(device))

    def forward(self, spectrogram: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """ Spectrograms shape: (batch_size, time, freq) """
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.unsqueeze(1)
        return self.encoder(spectrogram),


class AudioEmbeddingTagEncoder(Encoder):
    name = 'embedding-tag'

    def __init__(self, model_name: str = 'roberta-base', **kwargs):
        super().__init__(**kwargs)
        self.tag_encoder = TagEncoder(model_name=model_name)

    def infer(self, inputs: AudioCaptionOutput, device) -> Tuple[torch.Tensor, torch.Tensor]:
        tags_embeddings = self.tag_encoder.infer(inputs, device)
        return inputs.audio_embedding.to(device).unsqueeze(0), tags_embeddings

    def forward(self,
                audio_embedding: torch.Tensor,
                tag_input_ids: torch.Tensor,
                tag_attention_mask: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return audio_embedding, self.tag_encoder(tag_input_ids, tag_attention_mask)
