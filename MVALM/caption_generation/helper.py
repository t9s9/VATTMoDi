from typing import List, Union

import numpy as np
import torch
from transformers import GPT2Tokenizer


def _tokenize(s: str, tokenizer: GPT2Tokenizer):
    s = s.strip()
    if not s.endswith('.'):
        s += '.'

    return torch.tensor(tokenizer.encode(s), dtype=torch.int64)


def tokenize(s: Union[str, List[str]], tokenizer: GPT2Tokenizer):
    if isinstance(s, (list, np.ndarray)):
        return [_tokenize(sub, tokenizer) for sub in s]
    return _tokenize(s, tokenizer)


def pad(tokens, max_length):
    padding = max_length - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:max_length]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    mask = mask.float()
    return tokens, mask
