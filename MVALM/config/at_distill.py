import os
from multiprocessing import cpu_count
from pathlib import Path

import wandb
from sacred import Experiment
from transformers import CLIPVisionModel, CLIPTextModel

ex = Experiment("AT")


@ex.config
def config():
    """ basic configuration for all experiments """
    mode = 'train'

    id = wandb.util.generate_id(8)
    group = "AT-Full"
    project = "MoCo"

    beton_root = '/home/t9s9/Datasets/beton/'
    beton_file = 'VAT_audiocaps_declutr_{}.beton'

    checkpoint_path = '/home/t9s9/Datasets/ckpt/Distill'
    transformers_cache = None
    seed = 42
    debug = False
    overfit_batches = 0.0
    resume_from_checkpoint = None  # path to checkpoint

    # trainer
    max_epochs = 10
    precision = "16-mixed"
    sync_batchnorm = False
    gpus = [0]
    enable_progress_bar = True
    val_check_interval = 0.5
    limit_train_batches = None
    limit_val_batches = None
    do_test = False

    # logger
    logger = 'wandb'
    log_every_n_steps = 10

    # gradient
    gradient_clip_val = 1.0  # None to disable
    gradient_clip_algorithm = 'norm'  # 'value' or 'norm'

    # training
    training_kwargs = dict(
        distill=True,
        queue_size=16384 * 2,
        temp=0.07,
        momentum=0.995,
        alpha=0.4,
    )

    # captions
    caption_type = 'auditory'
    caption_mix_tag = 0.0

    if resume_from_checkpoint is not None:
        id = Path(resume_from_checkpoint).name[:8]

    exp_name = f"{id}"
    b = beton_file.format('').split('.')[0]

    # optimizer
    optimizer = dict(
        name='adamw',
        lr=1e-4,
        weight_decay=0.001,
        eps=1e-8,
        betas=[0.9, 0.98]
    )

    # scheduler
    lr_scheduler = dict(
        warmup_epochs=1,
        warmup_init_lr=1e-8,
        warmup_strategy='linear'
    )

    # callbacks
    callbacks = {
        'early_stopping': {
            'enabled': True,
            'params': {
                'monitor': 'val/total_loss',
                'patience': 3,
                'mode': 'min',
                'verbose': True,
                'min_delta': 0.0,
            }
        },
        'model_checkpoint': {
            'enabled': True,
            'params': {
                'save_last': False,
                'save_top_k': 2,
                'monitor': "val/total_loss",
                'mode': 'min',
                'verbose': True,
                'save_weights_only': False,
                'filename': f"{id}" + "-epoch={epoch}-step={step}-val_loss={val/total_loss:.3f}",
                'auto_insert_metric_name': False,
            }
        }
    }

    beton_path = os.path.join(beton_root, beton_file)
    workers = min(cpu_count(), 8)
    data = dict(
        train_beton=beton_path.format('train'),
        val_beton=beton_path.format('val'),
        batch_size=[256, 256, 256],
        num_workers=workers,
        drop_last=True,
        os_cache=True,
        ordering='quasi_random',
        batches_ahead=2,
        distributed=len(gpus) > 1,
        seed=seed,
    )

    model = dict(
        text_encoder=dict(
            model_name='johngiorgi/declutr-sci-base',
            # model_name='johngiorgi/declutr-small',
            max_length=60,
            avg_word_embs=False,  # use pooled output => eot token
            cache_dir=transformers_cache,
            gradient_checkpointing=True,
            # model_class=CLIPTextModel,
            freeze=False),
        audio_encoder=dict(
            model_name='passt_s_swa_p16_128_ap476',
            # model_name='htsat',
            s_patchout_t=40,  # 49
            s_patchout_f=4,  # 4
            gradient_checkpointing=True,
            freeze=False,
            cache_dir=transformers_cache),
        at=dict(
            proj_dim=512
        )
    )

    zero_shot = [
        # {'dataset_name': 'ESC50',
        #  'dataset': beton_root + 'evaluation/ESC50-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 16,
        #  'verbose': True,
        #  },
        # {'dataset_name': 'UrbanSound8k',
        #  'dataset': beton_root + 'evaluation/UrbanSound8k-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 16,
        #  'verbose': False,
        #  },
        # {'dataset_name': 'BDLib2',
        #  'dataset': beton_root + 'evaluation/BDLib2-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 16,
        #  'verbose': False,
        #  },
    ]

    exp_name += f"{b}{caption_type}"


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
def full():
    beton_file = 'VAT_full_declutr_{}.beton'
    val_check_interval = 1.0
    max_epochs = 20
    # limit_val_batches = 0

    group = "AT-Full"
    training_kwargs = dict(
        distill=False
    )


@ex.named_config
def audiocaps():
    beton_file = 'VAT_AudioCaps_declutr_{}.beton'
    val_check_interval = 1.0
    max_epochs = 20
    group = "AT-AudioCaps"

    training_kwargs = dict(
        distill=True
    )

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
def overfit():
    """ debug mode """
    debug = False
    do_test = False
    enable_progress_bar = True
    limit_train_batches = 50
    limit_val_batches = 50
    max_epochs = 1
    logger = 'tensorboard'
    group = "overfit"
    callbacks = {'model_checkpoint': {'enabled': False}}

    data = dict(
        batch_size=[256, 256, 256],
    )
