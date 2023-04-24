import os
from multiprocessing import cpu_count

import wandb
from sacred import Experiment

ex = Experiment("VAT")


@ex.config
def config():
    """ basic configuration for all experiments """
    mode = 'train'

    id = wandb.util.generate_id(8)
    exp_name = f"{id}"
    group = "VAT-Full-SimpleContrastiveLoss"
    project = "MoCo"

    beton_root = '/home/t9s9/Datasets/beton/'
    beton_file = 'VAT_audiocaps_declutr_{}.beton'

    checkpoint_path = '/home/t9s9/Datasets/ckpt'

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
    val_check_interval = 0.5
    do_test = False

    # logger
    logger = 'wandb'
    log_every_n_steps = 10

    # gradient
    gradient_clip_val = 1.0  # None to disable
    gradient_clip_algorithm = 'norm'  # 'value' or 'norm'

    # loss
    gather_features = True
    sync_grads = True

    # captions
    caption_type = 'auditory'

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
                'patience': 5,
                'mode': 'min',
                'verbose': True,
                'min_delta': 0.0,
            }
        },
        'model_checkpoint': {
            'enabled': True,
            'params': {
                'save_last': False,
                'save_top_k': -1,
                'monitor': "val/total_loss",
                'mode': 'min',
                'verbose': True,
                'save_weights_only': False,
                'filename': f"{id}" + "-epoch={epoch}-step={step}-val_loss={val/total_loss:.3f}",
                'auto_insert_metric_name': False,
            }
        }
    }

    beton_root = '/home/t9s9/Datasets/beton/'
    beton_file = 'VAT_full__declutr_{}.beton'

    beton_path = os.path.join(beton_root, beton_file)
    workers = min(cpu_count(), 8)
    data = dict(
        train_beton=beton_path.format('train'),
        val_beton=beton_path.format('val'),
        batch_size=[192, 192, 192],
        num_workers=workers,
        drop_last=True,
        os_cache=True,
        ordering='quasi_random',
        batches_ahead=2,
        distributed=len(gpus) > 1,
        seed=seed,
    )

    loss = dict(
        it=dict(temp=3.9, label_smoothing=0.0, train_temp=True),
        ia=dict(temp=3.9, label_smoothing=0.0, train_temp=True),
        ta=dict(temp=3.9, label_smoothing=0.0, train_temp=True),
    )

    model = dict(
        vision_encoder=dict(model_name='timm-vit_base_patch32_224_in21k',  # hf-openai/clip-vit-base-patch32,
                            # timm-swin_tiny_patch4_window7_224, timm-vit_base_patch32_224
                            cache_dir=transformers_cache,
                            gradient_checkpointing=True,
                            # model_class=CLIPVisionModel,
                            freeze=False),
        text_encoder=dict(model_name='johngiorgi/declutr-sci-base',
                          # openai/clip-vit-base-patch32, johngiorgi/declutr-sci-base
                          max_length=60,
                          avg_word_embs=False,  # use pooled output => eot token
                          cache_dir=transformers_cache,
                          gradient_checkpointing=True,
                          # model_class=CLIPTextModel,
                          freeze=False),
        audio_encoder=dict(model_name='passt_s_swa_p16_128_ap476',
                           s_patchout_t=40,  # 49
                           s_patchout_f=4,  # 4
                           gradient_checkpointing=True,
                           freeze=False),
        vat=dict(proj_dim=512, )
    )

    zero_shot = [
        # {'dataset_name': 'ESC50',
        #  'dataset': beton_root + 'evaluation/ESC50-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 32,
        #  },
        # {'dataset_name': 'UrbanSound8k',
        #  'dataset': beton_root + 'evaluation/UrbanSound8k-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 32,
        #  },
        # {'dataset_name': 'BDLib2',
        #  'dataset': beton_root + 'evaluation/BDLib2-spec.beton',
        #  'top_k': (1, 5),
        #  'perform_on_test': False,
        #  'perform_on_validation': True,
        #  'batch_size': 32,
        #  },
    ]

    b = beton_file.format('').split('.')[0]
    exp_name += f"SimpleContrastive_{b}{caption_type}"


@ex.named_config
def server():
    checkpoint_path = "/data/mmssl/ckpt/"
    beton_root = '/data/mmssl/beton/'
    transformers_cache = '/data/mmssl/cache'

    gpus = [0]


@ex.named_config
def debug():
    """ debug mode """
    debug = True
    enable_progress_bar = True


@ex.named_config
def full():
    beton_file = 'VAT_full_declutr_{}.beton'

    max_epochs = 20
    val_check_interval = 1.0

    exp_name = f"VAT_std_declutr_train"


@ex.named_config
def overfit():
    """ debug mode """
    debug = False
    do_test = False
    enable_progress_bar = True
    overfit_batches = 0.1
    max_epochs = 1
    logger = 'tensorboard'
    group = "overfit"
    callbacks = {'model_checkpoint': {'enabled': False}}
