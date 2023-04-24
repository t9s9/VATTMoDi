from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import AveragePrecision, Precision, Recall, Accuracy, ExactMatch

from MVALM.models.backbone.layers import Mlp
from MVALM.modules.lr_scheduler import WarmupCosineAnnealingLR
from MVALM.modules.optimizer import get_optimizer


class Classifier(pl.LightningModule):
    def __init__(self, model,
                 task: str,
                 dropout: float,
                 num_classes: int,
                 freeze_backbone: bool = False,
                 lr: float = 0.00001,
                 weight_decay: float = 0.0001,
                 warmup_epochs: int = 1,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__
        assert task in ('multiclass', 'multilabel')

        self.model = model
        self.head = Mlp(self.model.hidden_size, 2 * num_classes, num_classes,
                        bias=True, dropout=dropout)

        if task == 'multiclass':
            self.loss = nn.CrossEntropyLoss()
            metric_cls = [Accuracy]
        else:
            self.loss = nn.BCEWithLogitsLoss()
            metric_cls = [AveragePrecision, Precision, Recall, ExactMatch]

        metrics_kwargs = dict(task=task, num_classes=num_classes, num_labels=num_classes)
        self.metrics = nn.ModuleDict(
            {
                split: MetricCollection(*[cls(**metrics_kwargs) for cls in metric_cls], prefix=split[1:] + '_')
                for split in ('_train', '_val', '_test')
            }
        )

    def forward(self, x):
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            x = self.model(x)
        x = self.head(x)
        return x

    def on_fit_start(self) -> None:
        if self.hparams.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _step(self, batch, split):
        x, y = batch
        y = y.float()
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        metrics = self.metrics[split](y_hat, y)
        self.log(f'{split}/loss', loss, sync_dist=True, on_epoch=True, on_step=split == '_train', prog_bar=True)
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=split == '_train', prog_bar=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, '_train')

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, '_val')

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, '_test')

    def configure_optimizers(self):
        optimizer = get_optimizer("adam", self, dict(lr=self.hparams.lr, weight_decay=self.hparams.weight_decay),
                                  wd_ignore_layer_types=(nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d,),
                                  wd_ignore_parameter_names=('bias', 'logit_scale', 'loss_weight'))

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_epochs * (total_steps // self.trainer.max_epochs))

        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 max_steps=total_steps),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


