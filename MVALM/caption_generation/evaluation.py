import ast
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from fense.evaluator import Evaluator as Fense


def clean(s: Union[List[str], str]) -> Union[List[str], str]:
    """ Remove point and spaces from sentence """

    def _clean(s: str) -> str:
        s = s.strip().lower()
        if s.endswith('.'):
            s = s[:-1]
        return s

    if isinstance(s, (list, tuple, np.ndarray)):
        return [_clean(s_) for s_ in s]
    return _clean(s)


def score_auditory_captions(csv_file: str = None,
                            df=None,
                            preds: List[str] = None,
                            gts: List[List[str]] = None,
                            metric_prefix: Optional[Union[List[str], str]] = None,
                            bleu=True, rouge=True, cider=True, spice=True, meteor=True, fense=True, spider=True,
                            device=None) -> Dict:
    assert csv_file is not None or df is not None or (
            preds is not None and gts is not None), "Please provide either csv_file or df or (preds and gts)"
    metric_prefix = '-'.join(metric_prefix) if isinstance(metric_prefix, list) else metric_prefix

    metric_prefix = metric_prefix + '_' if metric_prefix is not None else ''
    result = {}

    if csv_file is not None:
        result['name'] = Path(csv_file).name
        df = pd.read_csv(csv_file)
        df['gt'] = df['gt'].apply(lambda x: ast.literal_eval(x))
    if df is not None:
        gts = df['gt'].tolist()
        preds = df['pred'].tolist()
    assert len(preds) == len(gts), "Number of predictions and ground truths do not match"

    if fense:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fense_scorer = Fense(sbert_model='paraphrase-TinyBERT-L6-v2', echecker_model='echecker_clotho_audiocaps_base',
                             batch_size=32, device=device)
        metric_name = f'{metric_prefix}FENSE' if metric_prefix is not None else 'FENSE'
        result[metric_name] = fense_scorer.corpus_score(cands=list(map(clean, preds)),
                                                        list_refs=list(map(clean, gts)),
                                                        agg_score='mean')

    scorers = []
    if cider:
        scorers.append((Cider(), "CIDEr"))
    if bleu:
        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
    if rouge:
        scorers.append((Rouge(), "ROUGE_L"))
    if spice:
        scorers.append((Spice(), "SPICE"))
    if meteor:
        scorers.append((Meteor(), "METEOR"))
    if not scorers:
        return result

    # convert to coco format
    gts = {i: [dict(caption=c.strip()) for c in gt] for i, gt in enumerate(gts)}
    preds = {i: [dict(caption=c.strip())] for i, c in enumerate(preds)}

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    preds = tokenizer.tokenize(preds)

    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, preds)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                metric_name = f'{metric_prefix}{m}'
                result[metric_name] = sc
        else:
            metric_name = f'{metric_prefix}{method}'
            result[metric_name] = score

    if cider and spice and spider:
        result[f'{metric_prefix}SPIDER'] = (result[f"{metric_prefix}SPICE"] + result[f"{metric_prefix}CIDEr"]) / 2

    return result


# Hugginface issue: https://github.com/huggingface/transformers/issues/6535
def beam_search(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.wte(tokens)
        for i in range(entry_length):
            outputs = model.caption_model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def greedy_search(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
        stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.wte(tokens)

            for i in range(entry_length):

                outputs = model.caption_model(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]
