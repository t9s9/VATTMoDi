import os
from multiprocessing import cpu_count
import wandb
from timm import list_models
from sacred import Experiment
from transformers import CLIPVisionModel, CLIPTextModel

ex = Experiment("AT")


@ex.config
def config():
    """ basic configuration for all experiments """
    mode = 'train'

    id = wandb.util.generate_id(8)
    exp_name = f"{id}"
    group = "AT-clip-mcap-ds-vgg"
    project = "AT"

    checkpoint_path = '/home/t9s9/Datasets/ckpt'
    transformers_cache = None
    seed = 42
    debug = False
    overfit_batches = 0.0
    resume_from_checkpoint = None  # path to checkpoint

    # trainer
    max_epochs = 20
    precision = 16
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
    track_grad_norm = -1  # -1 no tracking, 0 inf norm, 1 1-norm, 2 2-norm

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
                'patience': 6,
                'mode': 'min',
                'verbose': True,
                'min_delta': 0.0,
            }
        },
        'model_checkpoint': {
            'enabled': True,
            'params': {
                'save_last': True,
                'save_top_k': 5,
                'monitor': "val/total_loss",
                'mode': 'min',
                'verbose': True,
                'save_weights_only': False,
                'filename': f"{exp_name}" + "-epoch={epoch}-val_loss={val/total_loss:.3f}",
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
        batch_size=[256, 32, 32],
        num_workers=workers,
        drop_last=True,
        os_cache=False,
        ordering='quasi_random',
        batches_ahead=3,
        distributed=len(gpus) > 1,
        seed=seed,
    )

    loss = dict(temp=3.9, label_smoothing=0.0, train_temp=True)

    model = dict(
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
        at=dict(proj_dim=768, )
    )

    zero_shot = [
        {'dataset_name': 'ESC50',
         'dataset': beton_root + 'evaluation/ESC50-spec.beton',
         'top_k': (1, 5),
         'perform_on_test': False,
         'perform_on_validation': True,
         'batch_size': 32,
         },
        {'dataset_name': 'UrbanSound8k',
         'dataset': beton_root + 'evaluation/UrbanSound8k-spec.beton',
         'top_k': (1, 5),
         'perform_on_test': False,
         'perform_on_validation': True,
         'batch_size': 32,
         },
        {'dataset_name': 'BDLib2',
         'dataset': beton_root + 'evaluation/BDLib2-spec.beton',
         'top_k': (1, 5),
         'perform_on_test': False,
         'perform_on_validation': True,
         'batch_size': 32,
         },
    ]


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
    beton_file = 'VAT_full__declutr_{}.beton'
    max_epochs = 10
    exp_name = f"AT_full_declutr_train"
    logger = 'tensorboard'
    val_check_interval = 1 / 3
    caption_type = 'auditory'


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
