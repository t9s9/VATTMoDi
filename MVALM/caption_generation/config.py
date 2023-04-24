import wandb
from sacred import Experiment
import os
from model import PrexfixMappingType, AudioEmbeddingTagEncoder, SpectrogramEncoder, SingleEmbeddingEncoder

ex = Experiment("CaptioningModel")


@ex.config
def config():
    """ basic configuration for all experiments """
    mode = 'train'

    project = "AudioCaptioning"
    group = "audio_v2"
    id = wandb.util.generate_id(8)

    checkpoint_path = '/home/t9s9/Datasets/ckpt'
    transformers_cache = None
    seed = 42
    debug = False
    overfit_batches = 0.0
    resume_from_checkpoint = None  # path to checkpoint

    # trainer
    precision = "16-mixed"
    max_epochs = 20
    gpus = [0]
    enable_progress_bar = True
    val_check_interval = 1.0
    do_test = True
    test_root = {'MACS': '/media/t9s9/SSD_ubuntu/datasets',
                 'Clotho': '/media/t9s9/SSD_ubuntu/datasets',
                 'AudioCaps': '/home/t9s9/Datasets/'}

    # logger
    logger = 'wandb'
    log_every_n_steps = 20

    # model training parameter
    lr = 1e-05

    num_warmup_steps = 8000
    train_gpt = False

    # model
    model = dict(
        encoder=SingleEmbeddingEncoder,
        encoder_kwargs={'freeze': True,
                        's_patchout_t': 0,
                        's_patchout_f': 0,
                        'model_name': 'roberta-base'},
        input_dim=(768,),
        const_prefix_length=10,
        prefix_mapping_type=PrexfixMappingType.MLP,
        embedding_prefix_length=None,
        num_layers=None,
        num_heads=None,
        caption_model='gpt2',
    )

    # dataset
    dataset_root = '/home/t9s9/Datasets/beton/AudioCaptionGeneration/'
    dataset_name = 'CapGen_Audio_{}.beton'

    dataset_path = dataset_root + dataset_name
    data = dict(
        train_beton=dataset_path.format('train'),
        val_beton=dataset_path.format('val'),
        test_beton=None,
        batch_size=[64, 64, 0],
        num_workers=[8, 4, 0],
        drop_last=False,
        ordering='random',
        distributed=False,
        os_cache=True
    )

    exp_name = f"{id}_{model['prefix_mapping_type'].value}_{model['encoder'].name}_{'train_gpt' if train_gpt else 'frozen_gpt'}"

    callbacks = {
        'early_stopping': {
            'enabled': True,
            'params': {
                'monitor': 'val_loss',
                'patience': 2,
                'mode': 'min',
                'verbose': True,
                'min_delta': 0.02,
            }
        },
        'model_checkpoint': {
            'enabled': True,
            'params': {
                'save_last': False,
                'save_top_k': 1,
                'monitor': "val_loss",
                'mode': 'min',
                'verbose': True,
                'save_weights_only': False,
                'filename': f"{exp_name}" + "-{epoch}-{val_loss:.3f}"
            }
        }
    }

    if transformers_cache is not None:
        print("set transformers cache")
        os.environ['TRANSFORMERS_CACHE'] = str(transformers_cache)


@ex.named_config
def server():
    test_root = {'Clotho': '/data/mmssl/',
                 'AudioCaps': '/data/mmssl/'}
    checkpoint_path = "/data/mmssl/ckpt/Caption"
    dataset_root = '/data/mmssl/beton/AudioCap/'
    transformers_cache = '/data/mmssl/cache'


@ex.named_config
def debug():
    """ debug mode """
    debug = True
    enable_progress_bar = True


@ex.named_config
def overfit():
    """ debug mode """
    debug = False
    do_test = True
    enable_progress_bar = True
    overfit_batches = 0.1
    max_epochs = 1
    logger = 'wandb'
    group = "overfit"
    callbacks = {'model_checkpoint': {'enabled': True}}


@ex.named_config
def transformer():
    model = dict(
        input_dim=(768,),
        const_prefix_length=10,
        prefix_mapping_type=PrexfixMappingType.Transformer,
        embedding_prefix_length=(10,),
        num_layers=6,
        num_heads=12,
    )


@ex.named_config
def gpt_medium():
    group = "audio_gpt_medium"
    model = dict(language_model='gpt2-medium',
                 num_heads=8,  # embed_dim = 1024
                 num_layers=8)


@ex.named_config
def tag():
    dataset_name = 'CapGen_Audio+Tag_Roberta_{}.beton'
    group = "tag"

    train_gpt = False

    model = dict(
        encoder=AudioEmbeddingTagEncoder,
        encoder_kwargs={'freeze': False,
                        'model_name': 'roberta-base'},
        input_dim=(768, 768),
        const_prefix_length=10,
        prefix_mapping_type=PrexfixMappingType.MLP,
    )


@ex.named_config
def tag_transformer():
    dataset_name = 'CapGen_Audio+Tag_Roberta_{}.beton'
    group = "tag"

    train_gpt = False

    model = dict(
        encoder=AudioEmbeddingTagEncoder,
        encoder_kwargs={'freeze': False,
                        'model_name': 'roberta-base'},
        input_dim=(768, 768),
        const_prefix_length=10,
        prefix_mapping_type=PrexfixMappingType.Transformer,
        embedding_prefix_length=(10, 10),
        num_layers=4,
        num_heads=4
    )


@ex.named_config
def spectrogram():
    dataset_name = 'CapGen_Spectrogram_{}.beton'

    model = dict(
        encoder=SpectrogramEncoder,
        encoder_kwargs={'s_patchout_t': 33,
                        's_patchout_f': 4},
        input_dim=(768,),
        const_prefix_length=10,
        prefix_mapping_type=PrexfixMappingType.MLP,
    )


@ex.named_config
def train_caption():
    train_gpt = True


@ex.named_config
def opt():
    group = "opt"
    dataset_name = 'CapGen_OPT_Audio_{}.beton'
    model = dict(caption_model='facebook/opt-125m')
