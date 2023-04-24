0.2 import os
from multiprocessing import cpu_count
from pathlib import Path

import wandb
from sacred import Experiment

ex = Experiment("Finetune")


@ex.config
def config():
    id = wandb.util.generate_id(8)
    group = "VAT"
    project = "Finetune"

    beton_root = None
    beton_file = None

    checkpoint_path = '/home/t9s9/Datasets/ckpt/Finetune'
    transformers_cache = None
    seed = 42
    debug = False
    overfit_batches = 0.0
    resume_from_checkpoint = None  # path to checkpoint

    # trainer
    max_epochs = 20
    precision = "16-mixed"
    sync_batchnorm = False
    gpus = [0]
    enable_progress_bar = True
    val_check_interval = 1.0
    limit_train_batches = None
    limit_val_batches = None
    do_test = True

    # logger
    logger = 'wandb'
    log_every_n_steps = 10

    if resume_from_checkpoint is not None:
        id = Path(resume_from_checkpoint).name[:8]

    exp_name = f"{id}"
    b = beton_file.format('').split('.')[0]

    # callbacks
    callbacks = {
        'early_stopping': {
            'enabled': True,
            'params': {
                'monitor': '_val/loss',
                'patience': 2,
                'mode': 'min',
                'verbose': True,
                'min_delta': 0.0,
            }
        },
        'model_checkpoint': {
            'enabled': True,
            'params': {
                'save_last': False,
                'save_top_k': 1,
                'monitor': "_val/loss",
                'mode': 'min',
                'verbose': True,
                'save_weights_only': False,
                'filename': f"{id}" + "-epoch={epoch}-step={step}",
                'auto_insert_metric_name': False,
            }
        }
    }

    beton_path = os.path.join(beton_root, beton_file)
    workers = min(cpu_count(), 8)
    data = dict(
        train_beton=beton_path.format('train'),
        val_beton=beton_path.format('val'),
        test_beton=beton_path.format('test'),
        batch_size=[128, 16, 16],
        num_workers=workers,
        drop_last=True,
        os_cache=True,
        ordering='random',
        batches_ahead=3,
        distributed=len(gpus) > 1,
        seed=seed,
    )

    # classifier
    task = 'multiclass'
    dropout = 0.2
    num_classes = 10
    freeze_backbone = False
    lr = 1e-3
    weight_decay = 1e-4
    warmup_epochs = 0

    model = dict(
        path=None,
        momentum=False,
        gradient_checkpointing=not freeze_backbone,
    )

    exp_name += f"{b}_{'probe' if freeze_backbone else 'finetune'}_{Path(model['path']).name[:8]}"


@ex.named_config
def server():
    checkpoint_path = "/data/mmssl/ckpt/Distill"
    beton_root = '/data/mmssl/beton/'
    transformers_cache = '/data/mmssl/cache'

    gpus = [0, 1, 2, 3]

    data = dict(
        batch_size=[128, 128, 128],
        ordering='random',
    )


@ex.named_config
def debug():
    """ debug mode """
    debug = True
    enable_progress_bar = True


@ex.named_config
def freeze():
    """ freeze backbone """
    freeze_backbone = True


@ex.named_config
def fsd():
    beton_root = "/home/t9s9/Datasets/beton/evaluation/"
    beton_file = "FSD50k{}-spec.beton"

    task = 'multilabel'
    num_classes = 200
    group = "VAT-FSD"

    model = dict(
        path="/home/t9s9/Datasets/ckpt/VAT-Distill/65mghu8x/65mghu8x-epoch=3-step=3860-val_loss=10.900.ckpt",
        momentum=True,
        gradient_checkpointing=True,
    )

    max_epochs = 20
    freeze_backbone = False
    lr = 1e-3
    weight_decay = 1e-4
    warmup_epochs = 1
