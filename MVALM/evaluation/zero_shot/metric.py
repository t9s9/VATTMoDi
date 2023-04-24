from typing import Tuple, List, Dict
import numpy as np
import torch


def save_div(numerator, denominator):
    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    numerator = torch.where(zero_div_mask, torch.tensor(0.0, device=numerator.device), numerator)
    denominator = torch.where(zero_div_mask, torch.tensor(1.0, device=denominator.device), denominator)
    return numerator / denominator


class MetricList:
    def __init__(self, cls, number, metric_args):
        if not isinstance(metric_args, list):
            metric_args = [metric_args for _ in range(number)]
        self.metrics = [cls(**metric_args[i]) for i in range(number)]

    def __call__(self, pred, target, **kwargs):
        for metric in self.metrics:
            metric(pred, target, **kwargs)

    def compute(self):
        return [(str(metric), metric.compute()) for metric in self.metrics]

    def best(self, metric):
        result = self.compute()
        return result[np.array(list(map(lambda x: x[1][metric], result))).argmax()]

    def bar(self):
        return self.best('f1')


class MultiLabelConfusion:
    def __init__(self, threshold=0.5, classwise=False):
        self.result = {}
        self.threshold = threshold
        self.classwise = classwise
        self.reset()

    def __str__(self):
        return "MultiLabelConfusion (threshold={0:.3f})".format(self.threshold)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def bar(self):
        self.compute()
        return {f'{metric}': '{0:.3f}'.format(self.result[metric]) for metric in ['f1', 'precision', 'recall']}

    def update(self, pred, target):
        pred = (pred >= self.threshold).type_as(target)
        assert target.max() == 1, target
        assert pred.shape == target.shape
        assert pred.ndim == 2

        true_pred, false_pred = target == pred, target != pred
        pos_pred, neg_pred = pred == 1.0, pred == 0.0

        dim = 0 if self.classwise else [0, 1]
        self.tp += (true_pred * pos_pred).sum(dim=dim).float()
        self.fp += (false_pred * pos_pred).sum(dim=dim).float()

        self.tn += (true_pred * neg_pred).sum(dim=dim).float()
        self.fn += (false_pred * neg_pred).sum(dim=dim).float()

    def compute(self):
        result = {}
        total = (self.tp + self.tn + self.fp + self.fn)
        result['tp'] = self.tp / total
        result['fp'] = self.fp / total
        result['tn'] = self.tn / total
        result['fn'] = self.fn / total
        result['recall'] = save_div(self.tp, (self.tp + self.fn))
        result['precision'] = save_div(self.tp, (self.tp + self.fp))
        result['accuracy'] = (self.tp + self.tn) / total
        result['f1'] = 2 * save_div(result['precision'] * result['recall'], result['precision'] + result['recall'])
        self.result = result
        return result

    def __call__(self, pred, target, **kwargs):
        self.update(pred, target)


class Accuracy:
    def __init__(self, k: Tuple[int, ...]):
        self.k = k
        self.reset()

    def compute(self) -> Dict:
        return {str(k): round(avg, 5) for k, avg in zip(self.k, self.avg)}

    def bar(self):
        return {f'Top{k}': '{0:.3f}'.format(avg) for k, avg in zip(self.k, self.avg)}

    def reset(self):
        self.sum = np.zeros(len(self.k))
        self.count = np.zeros(len(self.k))
        self.avg = np.zeros(len(self.k))

    def update(self, output, target) -> List[float]:
        pred = output.topk(max(self.k), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in self.k]

    def __call__(self, output, target, n=1):
        acc = self.update(output, target)
        self.count += n
        self.sum += acc
        self.avg = self.sum / self.count
