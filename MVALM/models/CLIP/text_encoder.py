import torch.nn as nn
import torch
from ..backbone.transformer import Transformer
from ..backbone.layers import LayerNorm


class TextEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 gradient_checkpointing: bool = False,
                 save_attn: bool = False
                 ):
        super().__init__()
        self.context_length = context_length
        self.first_run = True

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            gradient_checkpointing=gradient_checkpointing,
            save_attn=save_attn
        )

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def initialize_parameters(self):
        self.transformer.initialize_parameters()
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def resize_token_embeddings(self, add_token_size: int):
        old_embeddings = self.token_embedding
        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape

        new_embeddings = nn.Embedding(old_num_tokens + add_token_size, old_embedding_dim)
        new_embeddings.to(
            old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[:old_num_tokens, :]
        self.token_embedding = new_embeddings
        print("Resized text embedding from {} to {}".format(old_num_tokens, old_num_tokens + add_token_size))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_feature(self, text):
        if self.first_run: print("TextEncoder")
        if self.first_run: print("Input text:", text.shape)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if self.first_run: print("Token embed:", x.shape)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.first_run: print("Before block", x.shape)
        x = self.transformer(x)
        if self.first_run: print("After block", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        if self.first_run: print("After ln_final", x.shape)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        if self.first_run: print("Out", x.shape)

        if self.first_run: self.first_run = False

        return x

    def encode_text(self, text):
        x = self.forward_feature(text)
        x = x @ self.text_projection
        return x


def base_text_transformers(**kwargs) -> TextEncoder:
    default_kwargs = {
        'embed_dim': 1024,
        'vocab_size': 49408,
        'context_length': 77,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    return TextEncoder(**default_kwargs)


def large_text_transformers(**kwargs) -> TextEncoder:
    default_kwargs = {
        'embed_dim': 768,
        'vocab_size': 49408,
        'context_length': 77,
        'transformer_width': 768,
        'transformer_heads': 12,
        'transformer_layers': 12,
        'gradient_checkpointing': False
    }
    default_kwargs.update(**kwargs)
    return TextEncoder(**default_kwargs)
