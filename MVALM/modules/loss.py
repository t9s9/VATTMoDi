import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss that supports label_smoothing and learnable temperature.
    """

    def __init__(self,
                 temp: float = 3.9,
                 label_smoothing: float = 0.0,
                 train_temp: bool = True):
        super().__init__()
        alpha = 1.0 - label_smoothing
        if alpha < 1.0:
            self.loss = SmoothContrastiveLoss(temp, train_temp, alpha)
        else:
            self.loss = HardContrastiveLoss(temp, train_temp)

    def forward(self, logits_per_a, logits_per_b, pl_module: pl.LightningModule):
        return self.loss(logits_per_a, logits_per_b, pl_module)

    @property
    def temp(self):
        return self.loss.temp


class HardContrastiveLoss(nn.Module):
    def __init__(self, temp: float = 3.9, train_temp: bool = True):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(temp, requires_grad=train_temp))

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, logits_per_a, logits_per_b, pl_module: pl.LightningModule):
        temp = torch.clamp(torch.exp(self.temp), min=1.0, max=100.0)

        num_logits = logits_per_a.shape[0]

        if self.prev_num_logits != num_logits or pl_module.device not in self.labels:
            labels = torch.arange(num_logits, device=pl_module.device, dtype=torch.long)
            labels += num_logits * pl_module.local_rank
            # cache state
            self.labels[pl_module.device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[pl_module.device]

        return (
                F.cross_entropy(temp * logits_per_a, labels, reduction="mean") +
                F.cross_entropy(temp * logits_per_b, labels, reduction="mean")
        ) / 2


class SmoothContrastiveLoss(nn.Module):
    def __init__(self,
                 temp: float = 3.9,
                 train_temp: bool = True,
                 alpha: float = 1.0,
                 ):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(temp, requires_grad=train_temp))
        assert alpha > 0 and alpha < 1
        self.alpha = alpha
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits_per_a, logits_per_b, pl_module: pl.LightningModule):
        temp = torch.clamp(torch.exp(self.T), min=1.0, max=100.0)
        pred_logprob = self.logsoftmax(logits * temp)

        with torch.no_grad():
            eps = (1 - self.alpha) / (logits.shape[1] - 1)
            batch = logits.shape[0]
            offset = pl_module.local_rank * batch

            t_prob = torch.ones(*logits.shape) * eps
            t_prob[:, offset:offset + batch] += torch.eye(batch) * (self.alpha - eps)
            t_prob = t_prob.to(logits)
            t_ent = (-t_prob * torch.log(t_prob)).sum(dim=1).mean()

        return self.kld(input=pred_logprob, target=t_prob) + t_ent


class KLDivLoss(nn.Module):
    """
    Wrapped KLDivergence Loss with a learnable temperature.
    """

    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.temp = nn.Parameter(torch.tensor(3.9, requires_grad=True))

    def forward(self, pred, target_prob):
        """
        Pred is logits and target is probabilities.
        """
        temp = torch.clamp(torch.exp(self.temp), min=1.0, max=100.0)
        pred_logprob = self.logsoftmax(pred * temp)

        return self.loss(input=pred_logprob, target=target_prob)
