from typing import NamedTuple, Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from MVALM.models.AT import AT, ATProj
from MVALM.models.VAT import VAT, VATProj
from MVALM.modules.loss import ContrastiveLoss
from MVALM.modules.lr_scheduler import WarmupCosineAnnealingLR
from MVALM.modules.optimizer import get_optimizer


class VATOutput(NamedTuple):
    total_loss: torch.Tensor
    loss_it: torch.Tensor
    loss_ia: torch.Tensor
    loss_ta: torch.Tensor
    acc_it: torch.Tensor
    acc_ia: torch.Tensor
    acc_ta: torch.Tensor


class ATModule(pl.LightningModule):
    def __init__(self,
                 model_kwargs: Dict,
                 optimizer_kwargs: Dict,
                 lr_scheduler_kwargs: Dict,
                 loss_kwargs: Dict,
                 caption_type: Union[str, float] = None,
                 gather_features: bool = True,
                 sync_grads: bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__
        self.model = AT.from_config(model_kwargs)

        self.loss = ContrastiveLoss(**loss_kwargs)

    def compute_logits(self, a_features, b_features, return_similarity=False):
        if self.hparams.gather_features and self.trainer.num_devices > 1:
            # result of all_gather is of shape (world_size, batch_size, ...)
            all_a_features = (
                self.all_gather(a_features, sync_grads=self.hparams.sync_grads)
                .view(-1, a_features.shape[-1])
            )
            all_b_features = (
                self.all_gather(b_features, sync_grads=self.hparams.sync_grads)
                .view(-1, b_features.shape[-1])
            )
        else:
            all_a_features = a_features
            all_b_features = b_features

        logits_per_a = a_features @ all_b_features.T
        logits_per_b = b_features @ all_a_features.T

        if return_similarity:
            sim_a = a_features @ all_a_features.T
            sim_b = b_features @ all_b_features.T
            return logits_per_a, logits_per_b, sim_a, sim_b

        return logits_per_a, logits_per_b

    @torch.no_grad()
    def accuracy(self, a_features, b_features):
        num_logits = a_features.shape[0]
        ground_truth = torch.arange(num_logits, device=self.device, dtype=torch.long)

        logits_per_a_metric = a_features @ b_features.T
        acc_per_a = (torch.argmax(logits_per_a_metric, 1) == ground_truth).sum()
        acc_per_b = (torch.argmax(logits_per_a_metric, 0) == ground_truth).sum()
        return (acc_per_a + acc_per_b) / 2 / num_logits

    def forward(self, input_ids=None, attention_mask=None, audio=None, spectrogram=None) -> ATProj:
        if audio is not None:
            with torch.cuda.amp.autocast(enabled=False):
                spec = self.model.audio_encoder.spectrogram(audio)
            audio = self.model.encode_audio(spec)
        elif spectrogram is not None:
            if spectrogram.ndim == 3:
                spectrogram = spectrogram.unsqueeze(1)
            audio = self.model.encode_audio(spectrogram)
        else:
            audio = None

        text = self.model.encode_text(input_ids, attention_mask) if (
                input_ids is not None and attention_mask is not None) else None

        return self.model.head(text, audio)

    def _step(self, batch):
        visual_input, audio_input, visual_caption, auditory_caption = batch  # caption shape (batch_size, 2, max_len)
        if self.hparams.caption_type == 'visual':
            input_ids, attention_mask = visual_caption.unbind(1)
        elif self.hparams.caption_type == 'auditory':
            input_ids, attention_mask = auditory_caption.unbind(1)
        elif isinstance(self.hparams.caption_type, float):
            if self.training:
                # if caption_type is a float, randomly choose between visual and auditory caption
                mask = torch.rand(visual_caption.shape[0], device=self.device) < self.hparams.caption_type
                visual_caption[mask] = auditory_caption[mask]
                input_ids, attention_mask = visual_caption.unbind(1)
            else:
                input_ids, attention_mask = auditory_caption.unbind(1)
        else:
            raise ValueError(f'Caption type {self.hparams.caption_type} is not supported')

        out: VATProj = self(input_ids, attention_mask, audio=audio_input)

        loss = self.loss(*self.compute_logits(out.t_proj, out.a_proj), pl_module=self)
        accuracy = self.accuracy(out.t_proj, out.a_proj)

        return loss, accuracy

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._step(train_batch)
        self.log('train/total_loss', loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train/accuracy', accuracy, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._step(val_batch)
        self.log('val/total_loss', loss, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/accuracy', accuracy, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_kwargs.pop('name')
        optimizer = get_optimizer(optimizer_name, self, self.hparams.optimizer_kwargs,
                                  wd_ignore_layer_types=(nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d,),
                                  wd_ignore_parameter_names=('bias', 'logit_scale', 'loss_weight')
                                  )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            self.hparams.lr_scheduler_kwargs.pop('warmup_epochs') * (total_steps // self.trainer.max_epochs))
        print(f"{total_steps=}, {warmup_steps=}, {self.trainer.max_epochs=}")

        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 max_steps=total_steps,
                                                 **self.hparams.lr_scheduler_kwargs),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


class VATModule(pl.LightningModule):
    def __init__(self,
                 model_kwargs: Dict,
                 optimizer_kwargs: Dict,
                 lr_scheduler_kwargs: Dict,
                 loss_kwargs: Dict,
                 caption_type: Union[str, float] = None,
                 gather_features: bool = True,
                 sync_grads: bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__
        self.model = VAT.from_config(model_kwargs)

        self.loss = nn.ModuleDict({
            'it': ContrastiveLoss(**loss_kwargs['it']),
            'ia': ContrastiveLoss(**loss_kwargs['ia']),
            'ta': ContrastiveLoss(**loss_kwargs['ta']),
        })

    def on_validation_start(self) -> None:
        # if PaSST is used we apply the patch masking during validation
        # to save memory
        if self.model.audio_encoder.model_name.startswith('passt'):
            self.model.audio_encoder.train()

    def compute_logits(self, a_features, b_features, return_similarity=False):
        if self.hparams.gather_features and self.trainer.num_devices > 1:
            # result of all_gather is of shape (world_size, batch_size, ...)
            all_a_features = (
                self.all_gather(a_features, sync_grads=self.hparams.sync_grads)
                .view(-1, a_features.shape[-1])
            )
            all_b_features = (
                self.all_gather(b_features, sync_grads=self.hparams.sync_grads)
                .view(-1, b_features.shape[-1])
            )
        else:
            all_a_features = a_features
            all_b_features = b_features

        logits_per_a = a_features @ all_b_features.T
        logits_per_b = b_features @ all_a_features.T

        if return_similarity:
            sim_a = a_features @ all_a_features.T
            sim_b = b_features @ all_b_features.T
            return logits_per_a, logits_per_b, sim_a, sim_b

        return logits_per_a, logits_per_b

    @torch.no_grad()
    def accuracy(self, a_features, b_features):
        num_logits = a_features.shape[0]
        ground_truth = torch.arange(num_logits, device=self.device, dtype=torch.long)

        logits_per_a_metric = a_features @ b_features.T
        acc_per_a = (torch.argmax(logits_per_a_metric, 1) == ground_truth).sum()
        acc_per_b = (torch.argmax(logits_per_a_metric, 0) == ground_truth).sum()
        return (acc_per_a + acc_per_b) / 2 / num_logits

    def forward(self, image=None, input_ids=None, attention_mask=None, audio=None, spectrogram=None) -> VATProj:
        return self.model(image=image, input_ids=input_ids, attention_mask=attention_mask, audio=audio,
                          spectrogram=spectrogram)

    def _step(self, batch) -> VATOutput:
        visual_input, audio_input, visual_caption, auditory_caption, tag_caption = batch
        if self.hparams.caption_type == 'visual':
            input_ids, attention_mask = visual_caption.unbind(1)
        elif self.hparams.caption_type == 'auditory':
            input_ids, attention_mask = auditory_caption.unbind(1)
        elif isinstance(self.hparams.caption_type, float):
            if self.training:
                # if caption_type is a float, randomly choose between visual and auditory caption
                mask = torch.rand(visual_caption.shape[0], device=self.device) < self.hparams.caption_type
                visual_caption[mask] = auditory_caption[mask]
                input_ids, attention_mask = visual_caption.unbind(1)
            else:
                input_ids, attention_mask = auditory_caption.unbind(1)
        else:
            raise ValueError(f'Caption type {self.hparams.caption_type} is not supported')

        out: VATProj = self(visual_input, input_ids, attention_mask, audio=audio_input)

        loss_ta = self.loss['ta'](*self.compute_logits(out.t_proj, out.a_proj), pl_module=self)
        loss_ia = self.loss['ia'](*self.compute_logits(out.i_proj, out.a_proj), pl_module=self)
        loss_it = self.loss['it'](*self.compute_logits(out.i_proj, out.t_proj), pl_module=self)

        total_loss = (loss_ia + loss_ta + loss_it) / 3

        acc_ta = self.accuracy(out.t_proj, out.a_proj)
        acc_ia = self.accuracy(out.i_proj, out.a_proj)
        acc_it = self.accuracy(out.i_proj, out.t_proj)

        return VATOutput(total_loss=total_loss,
                         loss_it=loss_it, loss_ia=loss_ia, loss_ta=loss_ta,
                         acc_it=acc_it, acc_ia=acc_ia, acc_ta=acc_ta)

    def training_step(self, train_batch, batch_idx):
        output: VATOutput = self._step(train_batch)
        data = {('train/' + key): value for key, value in dict(output._asdict()).items()}
        self.log_dict(data, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        return output.total_loss

    def validation_step(self, val_batch, batch_idx):
        output: VATOutput = self._step(val_batch)
        data = {(f'val/' + key): value for key, value in dict(output._asdict()).items()}
        self.log_dict(data, sync_dist=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        output: VATOutput = self._step(test_batch)
        data = {(f'test/' + key): value for key, value in dict(output._asdict()).items()}
        self.log_dict(data, sync_dist=True)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_kwargs.pop('name')
        optimizer = get_optimizer(optimizer_name, self, self.hparams.optimizer_kwargs,
                                  wd_ignore_layer_types=(nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d,),
                                  wd_ignore_parameter_names=('bias', 'logit_scale', 'loss_weight')
                                  )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            self.hparams.lr_scheduler_kwargs.pop('warmup_epochs') * (total_steps // self.trainer.max_epochs))
        print(f"{total_steps=}, {warmup_steps=}, {self.trainer.max_epochs=}")

        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 max_steps=total_steps,
                                                 **self.hparams.lr_scheduler_kwargs),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
