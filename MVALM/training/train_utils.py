import os
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import wandb
from pytorch_lightning import seed_everything
from torch.cuda import empty_cache


def prepare_workspace(_config: Dict) -> Dict:
    config = deepcopy(_config)

    checkpoint_dir = Path(config['checkpoint_path'])
    checkpoint_dir.mkdir(exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
    empty_cache()

    seed_everything(config['seed'])

    torch.set_float32_matmul_precision('medium')

    return config


def finalize_workspace():
    empty_cache()
    wandb.finish()
