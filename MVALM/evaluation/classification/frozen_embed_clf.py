from typing import List

import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
from sklearn.metrics import accuracy_score
from MVALM.datasets import ESC50
from tqdm import tqdm
from MVALM.evaluation.utils import load_model, eval_dataset
import pandas as pd
from pathlib import Path


@torch.no_grad()
def feature_extraction(module, dataset, momentum, device):
    features = []
    labels = []
    if momentum:
        m = module.model_m
    else:
        m = module.model

    for spectrograms, targets in tqdm(dataset):
        spectrograms = spectrograms.to(device).half()
        pred = m.encode_audio(spectrogram=spectrograms)
        features.append(pred.cpu())
        labels.append(targets.clone())

    features = torch.cat(features).float()
    labels = torch.cat(labels).view(-1)
    return features.numpy(), labels.numpy()


def fit_regression(X, y, cv=5, Cs=16):
    model = LogisticRegressionCV(penalty='l2',
                                 max_iter=1000,
                                 Cs=Cs,
                                 cv=cv,  # cv_sampler,
                                 random_state=0,
                                 solver='lbfgs',
                                 multi_class='multinomial',
                                 scoring='accuracy',
                                 n_jobs=-1)

    model.fit(X, y)
    result = model.scores_[0]
    # average over folds
    folds = result.mean(axis=0)
    folds_std = result.std(axis=0)

    # take max value over Cs
    best_core = folds.max(0)

    best_score_std = folds_std[folds.argmax(0)]

    best_c = model.C_[0]

    return {'accuracy': best_core, 'C': best_c, 'std': best_score_std}


def frz_clf_run(ckpt: str,
                datasets: List[str],
                momentum=False,
                device='cuda'
                ) -> pd.DataFrame:
    result = []
    module = load_model(ckpt, device)

    if isinstance(momentum, bool):
        momentum = [momentum]
    else:
        momentum = [False, True]
    if not hasattr(module, 'model_m'):
        momentum = [False]

    for dataset_name in datasets:
        dataset, _ = eval_dataset(dataset_name, batch_size=8)
        for mom in momentum:
            X, y = feature_extraction(module, dataset, momentum=mom, device=device)
            res = fit_regression(X, y)
            res['dataset'] = dataset_name
            res['momentum'] = mom
            res['ckpt'] = Path(ckpt).name
            result.append(res)

    return pd.DataFrame(result)


if __name__ == '__main__':
    # ckpt = "/home/t9s9/Datasets/ckpt/AT-Distill/qi4hnpdd/qi4hnpdd-epoch=6-val_loss=1.275.ckpt"
    # ckpt = "/home/t9s9/Datasets/ckpt/VAT-Distill/9ggaddyl/9ggaddyl-epoch=6-step=13517-val_loss=18.984.ckpt"

    ckpt = "/home/t9s9/Datasets/ckpt/VAT-Distill/65mghu8x/65mghu8x-epoch=3-step=3860-val_loss=10.900.ckpt"
    device = "cuda"

    print(frz_clf_run(ckpt, ['ESC50'], momentum='both'))


    # cv_sampler = PredefinedSplit(test_fold=dataset.sample_to_5fold())



