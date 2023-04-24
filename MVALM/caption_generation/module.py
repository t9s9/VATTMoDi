import time
from copy import deepcopy
from itertools import chain
from typing import Union, Tuple, List, Iterator

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, GPT2Tokenizer

from evaluation import beam_search, greedy_search, score_auditory_captions
from model import CaptioningModelHead, PrexfixMappingType, SingleEmbeddingEncoder
from dataset import AudioCaptionOutput


class CaptioningModule(LightningModule):
    def __init__(self,
                 input_dim: Tuple[int, ...],
                 const_prefix_length: int,
                 prefix_mapping_type: PrexfixMappingType = PrexfixMappingType.MLP,
                 embedding_prefix_length: Tuple[int, ...] = None,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 caption_model: str = 'gpt2',
                 encoder=None,
                 encoder_kwargs={},
                 lr: float = 1e-4,
                 num_warmup_steps: int = 100,
                 train_gpt: bool = False,
                 test_kwargs=None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.train_gpt = train_gpt
        self.test_kwargs = dict(strategy='beam', top_p=0.8, beam_size=5, max_length=60, temperature=1.0,
                                bleu=True, rouge=True, cider=True, spice=True, meteor=True, fense=True,
                                ) if test_kwargs is None else test_kwargs

        self.tokenizer = GPT2Tokenizer.from_pretrained(caption_model)

        self.encoder = SingleEmbeddingEncoder() if encoder is None else encoder(**encoder_kwargs)
        self.head = CaptioningModelHead(input_dim=input_dim,
                                        const_prefix_length=const_prefix_length,
                                        prefix_mapping_type=prefix_mapping_type,
                                        embedding_prefix_length=embedding_prefix_length,
                                        num_layers=num_layers,
                                        num_heads=num_heads,
                                        caption_model=caption_model)

        self.test_step_outputs = []

    def _step(self, batch):
        inputs = batch[:-2]

        embeddings = self.encoder(*inputs)

        # last two elements are gt captions and their masks
        tokens, masks = batch[-2:]
        outputs = self.head(embeddings=embeddings, tokens=tokens, mask=masks)

        logits = outputs.logits[:, self.hparams.const_prefix_length - 1: -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        return loss

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        if self.train_gpt:
            params = self.head.parameters(recurse=recurse)
        else:
            params = self.head.prefix_proj.parameters(recurse=recurse)

        if not self.hparams.encoder_kwargs.get('freeze', False):
            params = chain(params, self.encoder.parameters(recurse=recurse))

        return params

    def on_train_start(self) -> None:
        if self.hparams.encoder_kwargs.get('freeze', False):
            self.encoder.freeze()

    def on_train_epoch_start(self) -> None:
        if not self.train_gpt:
            self.head.caption_model.eval()
        if self.hparams.encoder_kwargs.get('freeze', False):
            self.encoder.eval()

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log('train_loss', loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=False)

    def test_step(self, batch: AudioCaptionOutput, batch_idx) -> Tuple[Union[str, List[str]], str]:
        pred = self.infer(batch, **self.test_kwargs)
        self.test_step_outputs.append((pred, batch.caption))
        return pred, batch.caption

    @torch.no_grad()
    def infer(self,
              inputs: AudioCaptionOutput,
              strategy: str = 'beam',
              beam_size: int = 5,
              max_length: int = 70,
              temperature: float = 1.0,
              top_p: float = 0.8,
              return_prefix: bool = False,
              **kwargs) -> Union[str, Tuple[str, torch.Tensor]]:
        embeddings = self.encoder.infer(inputs, device=self.device)
        prefix_embed = self.head.project_prefix(embeddings)

        if strategy == 'beam':
            pred = beam_search(self.head, self.tokenizer, embed=prefix_embed, beam_size=beam_size,
                               entry_length=max_length, temperature=temperature)[0]
        else:
            pred = greedy_search(self.head, self.tokenizer, embed=prefix_embed, entry_length=max_length,
                                 temperature=temperature, top_p=top_p)
        if return_prefix:
            return pred, prefix_embed

        return pred

    def on_test_epoch_end(self) -> None:
        t1 = time.time()
        preds, target = zip(*self.test_step_outputs)
        dataset_name = self.trainer.test_dataloaders.dataset.name if hasattr(
            self.trainer.test_dataloaders.dataset, 'name') else None
        scores = score_auditory_captions(preds=preds, gts=deepcopy(target), metric_prefix=dataset_name,
                                         bleu=self.test_kwargs['bleu'], rouge=self.test_kwargs['rouge'],
                                         cider=self.test_kwargs['cider'], spice=self.test_kwargs['spice'],
                                         meteor=self.test_kwargs['meteor'], fense=self.test_kwargs['fense'])
        print(f"Test took {time.time() - t1:.2f} seconds")

        self.test_step_outputs.clear()
        self.log_dict(scores)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, eps=1e-6)

        nr_batches = len(self.trainer.datamodule.train_dataloader())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=self.num_warmup_steps,
                                                       num_training_steps=self.trainer.max_epochs * nr_batches)
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
