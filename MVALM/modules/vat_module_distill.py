from copy import deepcopy
from typing import NamedTuple, Dict, Union, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from MVALM.models.AT import AT, ATProj
from MVALM.models.VAT import VAT, VATProj
from MVALM.modules.lr_scheduler import WarmupCosineAnnealingLR
from MVALM.modules.optimizer import get_optimizer
from MVALM.modules.helper import contrastive_accuracy


class DistillTrainingOptions(NamedTuple):
    distill: bool
    queue_size: int
    momentum: float
    alpha: float
    temp: float


class DistillModule(pl.LightningModule):
    def __init__(self,
                 optimizer_kwargs: Dict,
                 lr_scheduler_kwargs: Dict,
                 sync_grads: bool = False, ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__

    def resolve_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 3:
            audio_input, auditory_caption, tag_caption = batch  # caption shape (batch_size, 2, max_len)
            vision_input = None
        else:
            vision_input, audio_input, visual_caption, auditory_caption, tag_caption = batch

        if self.hparams.caption_type == 'auditory':
            if self.hparams.caption_mix_tag > 0.0:
                # use tag caption with 50% probability or auditory caption
                mask = torch.rand(auditory_caption.shape[0], device=self.device) < self.hparams.caption_mix_tag
                auditory_caption[mask] = tag_caption[mask]
            input_ids, attention_mask = auditory_caption.unbind(1)
        elif self.hparams.caption_type == 'visual':
            input_ids, attention_mask = visual_caption.unbind(1)
        elif isinstance(self.hparams.caption_type, float):
            if self.hparams.caption_mix_tag > 0.0:
                mask = torch.rand(auditory_caption.shape[0], device=self.device) < self.hparams.caption_mix_tag
                auditory_caption[mask] = tag_caption[mask]
            # if caption_type is a float, randomly choose between visual and auditory caption
            mask = torch.rand(visual_caption.shape[0], device=self.device) < self.hparams.caption_type
            visual_caption[mask] = auditory_caption[mask]
            input_ids, attention_mask = visual_caption.unbind(1)
        else:
            raise ValueError(f'Caption type {self.hparams.caption_type} is not supported')
        return input_ids, attention_mask, audio_input, vision_input

    def compute_logits(self, feat_a: torch.Tensor, feat_b: torch.Tensor, temp: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if self.trainer.num_devices > 1:
            # result of all_gather is of shape (world_size, batch_size, ...)
            all_feat_a = (
                self.all_gather(feat_a, sync_grads=self.hparams.sync_grads)
                .view(-1, feat_a.shape[-1])
            )
            all_feat_b = (
                self.all_gather(feat_b, sync_grads=self.hparams.sync_grads)
                .view(-1, feat_b.shape[-1])
            )
        else:
            all_feat_a = feat_a
            all_feat_b = feat_b

        logits_per_a = feat_a @ all_feat_b.T / temp
        logits_per_b = feat_b @ all_feat_a.T / temp

        return logits_per_a, logits_per_b

    def contrastive_loss(self, logits_per_a: torch.Tensor, logits_per_b: torch.Tensor) -> torch.Tensor:
        num_logits = logits_per_a.shape[0]
        labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
        labels += num_logits * self.local_rank
        return (
                F.cross_entropy(logits_per_a, labels, reduction="mean") +
                F.cross_entropy(logits_per_b, labels, reduction="mean")
        ) / 2

    def distill_loss(self, a, a_m, a_q, b, b_m, b_q, temp, alpha, sim_targets, distill: False):
        if distill:
            with torch.no_grad():
                sim_a2b_m = a_m @ b_q / temp
                sim_b2a_m = b_m @ a_q / temp

                sim_a2b_targets = alpha * F.softmax(sim_a2b_m, dim=1) + (1 - alpha) * sim_targets
                sim_b2a_targets = alpha * F.softmax(sim_b2a_m, dim=1) + (1 - alpha) * sim_targets

        sim_a2b = a @ b_q / temp
        sim_b2a = b @ a_q / temp

        if distill:
            loss_a2b = F.cross_entropy(sim_a2b, sim_a2b_targets, reduction="mean")
            loss_b2a = F.cross_entropy(sim_b2a, sim_b2a_targets, reduction="mean")
        else:
            loss_a2b = F.cross_entropy(sim_a2b, sim_targets, reduction="mean")
            loss_b2a = F.cross_entropy(sim_b2a, sim_targets, reduction="mean")

        return (loss_a2b + loss_b2a) / 2

    def on_validation_start(self) -> None:
        # if PaSST is used we apply the patch masking during validation
        # to save memory
        if self.model.audio_encoder.model_name.startswith('passt'):
            self.model.audio_encoder.train()

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

    @torch.no_grad()
    def _momentum_update(self):
        for param, param_m in zip(self.model.parameters(), self.model_m.parameters()):
            param_m.sub_((1. - self.training_kwargs.momentum) * (param_m - param))


class ATModule(DistillModule):
    def __init__(self,
                 training_kwargs: DistillTrainingOptions,
                 model_kwargs: Dict,
                 optimizer_kwargs: Dict,
                 lr_scheduler_kwargs: Dict,
                 caption_type: Union[str, float] = None,
                 caption_mix_tag: float = 0.0,
                 ):
        super().__init__(optimizer_kwargs=optimizer_kwargs, lr_scheduler_kwargs=lr_scheduler_kwargs)
        self.training_kwargs = training_kwargs
        self.caption_type = caption_type
        self.caption_mix_tag = caption_mix_tag
        self.temp = nn.Parameter(torch.ones([]) * training_kwargs.temp)

        self.model = AT.from_config(model_kwargs)
        self.model_m = deepcopy(self.model)

        for p in self.model_m.parameters():
            p.requires_grad = False

        # Queue
        self.register_buffer("audio_queue", torch.randn(self.model.proj_dim, self.training_kwargs.queue_size))
        self.register_buffer("text_queue", torch.randn(self.model.proj_dim, self.training_kwargs.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.audio_queue = F.normalize(self.audio_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

        # cache target
        self.sim_targets = {}
        self.prev_bs = 0

    def forward(self, input_ids=None, attention_mask=None, audio=None, spectrogram=None) -> ATProj:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, audio=audio, spectrogram=spectrogram)

    def training_step(self, train_batch, batch_idx):
        if self.current_epoch == 0:
            # warmup for 1 epoch
            alpha = self.training_kwargs.alpha * min(1, batch_idx / len(self.trainer.train_dataloader))
        else:
            alpha = self.training_kwargs.alpha

        input_ids, attention_mask, audio_input, _ = self.resolve_batch(train_batch)
        proj: ATProj = self(input_ids=input_ids, attention_mask=attention_mask, audio=audio_input)
        audio_feat = F.normalize(proj.a_proj, dim=-1)  # (batch_size, proj_dim)
        text_feat = F.normalize(proj.t_proj, dim=-1)

        batch_size = audio_feat.shape[0]

        self.log("train/accuracy", contrastive_accuracy(audio_feat, text_feat),
                 sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

            # update momentum model
            self._momentum_update()

            # get momentum feature
            proj_m: ATProj = self.model_m(input_ids=input_ids, attention_mask=attention_mask, audio=audio_input)
            audio_feat_m = F.normalize(proj_m.a_proj, dim=-1)  # (batch_size, proj_dim)
            text_feat_m = F.normalize(proj_m.t_proj, dim=-1)

            # combine feature with queue
            # (proj_dim, batch_size + queue_size)
            audio_feat_all = torch.cat([audio_feat_m.t(), self.audio_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # cache similarity targets
            if self.prev_bs != batch_size or self.device not in self.sim_targets:
                sim_targets = (
                    torch.zeros(size=(batch_size, batch_size + self.training_kwargs.queue_size))
                    .to(self.device)
                    .fill_diagonal_(1)
                )
                self.sim_targets[self.device] = sim_targets
                self.prev_bs = batch_size
            else:
                sim_targets = self.sim_targets[self.device]

        loss = self.distill_loss(audio_feat, audio_feat_m, audio_feat_all, text_feat, text_feat_m, text_feat_all,
                                 temp=self.temp, sim_targets=sim_targets, alpha=alpha,
                                 distill=self.training_kwargs.distill)

        # update queue
        self._dequeue_and_enqueue(audio_feat_m, text_feat_m)

        self.log('train/total_loss', loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, audio_feat, text_feat):
        # gather keys before updating queue
        # result of all_gather is of shape (world_size, batch_size, ...)
        audio_feats = self.all_gather(audio_feat, sync_grads=False).view(-1, audio_feat.shape[-1])
        text_feats = self.all_gather(text_feat, sync_grads=False).view(-1, text_feat.shape[-1])

        batch_size = audio_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.training_kwargs.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.training_kwargs.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def validation_step(self, val_batch, batch_idx):
        input_ids, attention_mask, audio_input, _ = self.resolve_batch(val_batch)
        proj: ATProj = self(input_ids=input_ids, attention_mask=attention_mask, audio=audio_input)
        a_proj = F.normalize(proj.a_proj, dim=-1)
        t_proj = F.normalize(proj.t_proj, dim=-1)

        logits_per_a, logits_per_b = self.compute_logits(a_proj, t_proj, temp=self.temp)
        loss = self.contrastive_loss(logits_per_a, logits_per_b)

        accuracy = contrastive_accuracy(a_proj, t_proj)

        self.log('val/total_loss', loss, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/accuracy', accuracy, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True)
        return loss


class VATModule(DistillModule):
    def __init__(self,
                 training_kwargs: DistillTrainingOptions,
                 model_kwargs: Dict,
                 optimizer_kwargs: Dict,
                 lr_scheduler_kwargs: Dict,
                 caption_type: Union[str, float] = None,
                 caption_mix_tag: float = 0.0,
                 ):
        super().__init__(optimizer_kwargs=optimizer_kwargs, lr_scheduler_kwargs=lr_scheduler_kwargs)
        self.training_kwargs = training_kwargs
        self.caption_type = caption_type
        self.caption_mix_tag = caption_mix_tag

        self.temp_at = nn.Parameter(torch.ones([]) * training_kwargs.temp)
        self.temp_ai = nn.Parameter(torch.ones([]) * training_kwargs.temp)
        self.temp_ti = nn.Parameter(torch.ones([]) * training_kwargs.temp)

        self.model = VAT.from_config(model_kwargs)
        self.model_m = deepcopy(self.model)

        for p in self.model_m.parameters():
            p.requires_grad = False

        # Queue
        self.register_buffer("audio_queue", torch.randn(self.model.proj_dim, self.training_kwargs.queue_size))
        self.register_buffer("image_queue", torch.randn(self.model.proj_dim, self.training_kwargs.queue_size))
        self.register_buffer("text_queue", torch.randn(self.model.proj_dim, self.training_kwargs.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.audio_queue = F.normalize(self.audio_queue, dim=0)
        self.vision_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

        # cache target
        self.sim_targets = {}
        self.prev_bs = 0

    def forward(self, image=None, input_ids=None, attention_mask=None, audio=None, spectrogram=None) -> VATProj:
        return self.model(image=image, input_ids=input_ids, attention_mask=attention_mask, audio=audio,
                          spectrogram=spectrogram)

    def training_step(self, train_batch, batch_idx):
        if self.current_epoch == 0:
            # warmup for 1 epoch
            alpha = self.training_kwargs.alpha * min(1, batch_idx / len(self.trainer.train_dataloader))
        else:
            alpha = self.training_kwargs.alpha

        input_ids, attention_mask, audio_input, vision_input = self.resolve_batch(train_batch)
        proj: VATProj = self(image=vision_input, input_ids=input_ids, attention_mask=attention_mask, audio=audio_input)
        audio_feat = F.normalize(proj.a_proj, dim=-1)
        image_feat = F.normalize(proj.i_proj, dim=-1)
        text_feat = F.normalize(proj.t_proj, dim=-1)

        batch_size = audio_feat.shape[0]

        self.log_dict({"train/accuracy_at": contrastive_accuracy(audio_feat, text_feat),
                       "train/accuracy_ai": contrastive_accuracy(audio_feat, image_feat),
                       "train/accuracy_ti": contrastive_accuracy(text_feat, image_feat)},
                      sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)

        with torch.no_grad():
            self.temp_ai.clamp_(0.001, 0.5)
            self.temp_ti.clamp_(0.001, 0.5)
            self.temp_at.clamp_(0.001, 0.5)

            # update momentum model
            self._momentum_update()

            # get momentum feature
            proj_m: VATProj = self.model_m(image=vision_input, input_ids=input_ids, attention_mask=attention_mask,
                                           audio=audio_input)
            audio_feat_m = F.normalize(proj_m.a_proj, dim=-1)
            image_feat_m = F.normalize(proj_m.i_proj, dim=-1)
            text_feat_m = F.normalize(proj_m.t_proj, dim=-1)

            # combine feature with queue
            audio_feat_all = torch.cat([audio_feat_m.t(), self.audio_queue.clone().detach()], dim=1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # cache similarity targets
            if self.prev_bs != batch_size or self.device not in self.sim_targets:
                sim_targets = (
                    torch.zeros(size=(batch_size, batch_size + self.training_kwargs.queue_size))
                    .to(self.device)
                    .fill_diagonal_(1)
                )
                self.sim_targets[self.device] = sim_targets
                self.prev_bs = batch_size
            else:
                sim_targets = self.sim_targets[self.device]

        loss_at = self.distill_loss(audio_feat, audio_feat_m, audio_feat_all, text_feat, text_feat_m, text_feat_all,
                                    temp=self.temp_at, alpha=alpha, sim_targets=sim_targets,
                                    distill=self.training_kwargs.distill)
        loss_ti = self.distill_loss(text_feat, text_feat_m, text_feat_all, image_feat, image_feat_m, image_feat_all,
                                    temp=self.temp_ti, alpha=alpha, sim_targets=sim_targets,
                                    distill=self.training_kwargs.distill)
        loss_ai = self.distill_loss(audio_feat, audio_feat_m, audio_feat_all, image_feat, image_feat_m, image_feat_all,
                                    temp=self.temp_ai, alpha=alpha, sim_targets=sim_targets,
                                    distill=self.training_kwargs.distill)
        loss = loss_at + loss_ti + loss_ai

        # update queue
        self._dequeue_and_enqueue(image_feat_m, audio_feat_m, text_feat_m)

        self.log_dict({'train/loss_at': loss_at, 'train/loss_a2i': loss_ai, 'train/loss_ti': loss_ti,
                       'train/total_loss': loss},
                      sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, audio_feat, text_feat):
        # gather keys before updating queue
        # result of all_gather is of shape (world_size, batch_size, ...)
        audio_feats = self.all_gather(audio_feat, sync_grads=False).view(-1, audio_feat.shape[-1])
        image_feats = self.all_gather(image_feat, sync_grads=False).view(-1, image_feat.shape[-1])
        text_feats = self.all_gather(text_feat, sync_grads=False).view(-1, text_feat.shape[-1])

        batch_size = audio_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.training_kwargs.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feats.T
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.training_kwargs.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def validation_step(self, val_batch, batch_idx):
        input_ids, attention_mask, audio_input, vision_input = self.resolve_batch(val_batch)
        proj: VATProj = self(input_ids=input_ids, attention_mask=attention_mask, audio=audio_input, image=vision_input)
        a_proj = F.normalize(proj.a_proj, dim=-1)
        t_proj = F.normalize(proj.t_proj, dim=-1)
        i_proj = F.normalize(proj.i_proj, dim=-1)

        loss_at = self.contrastive_loss(*self.compute_logits(a_proj, t_proj, temp=self.temp_at))
        loss_ai = self.contrastive_loss(*self.compute_logits(a_proj, i_proj, temp=self.temp_ai))
        loss_ti = self.contrastive_loss(*self.compute_logits(t_proj, i_proj, temp=self.temp_ti))
        loss = loss_at + loss_ai + loss_ti

        accuracy_at = contrastive_accuracy(a_proj, t_proj)
        accuracy_ai = contrastive_accuracy(a_proj, i_proj)
        accuracy_ti = contrastive_accuracy(t_proj, i_proj)

        self.log_dict({'val/loss_at': loss_at, 'val/loss_ai': loss_ai, 'val/loss_ti': loss_ti, 'val/total_loss': loss},
                      sync_dist=True, on_epoch=True, on_step=False)
        self.log_dict({'val/accuracy_at': accuracy_at, 'val/accuracy_ai': accuracy_ai, 'val/accuracy_ti': accuracy_ti},
                      sync_dist=True, on_epoch=True, on_step=False)
        return loss
