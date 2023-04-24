import sys
from copy import deepcopy

sys.path.append('/home/t9s9/PycharmProjects/Multimodal-VAL-Models')

from pathlib import Path

from sacred import Experiment

import MVALM.datasets as ds
from MVALM.clean_generation.dataset import (AudioCaptionPostprocessor,
                                            AudioCapDatasetWrapper,
                                            AudioTagCaptionPostprocessor,
                                            AudioSpectrogramCaptionPostprocessor,
                                            create_datasets,
                                            create_audio_beton)

ex = Experiment('AudioCapDataset')


@ex.config
def config():
    root = '/media/t9s9/SSD_ubuntu/datasets/'

    datasets = {
        'AudioCaps': ('/home/t9s9/Datasets/', ds.AudioCapsAudioOnly),
        'MACS': (root, ds.MACS),
        'Clotho': (root, ds.Clotho),
        'WavText5k': (root, ds.WavText5k),
        'SoundDescs': (root, ds.SoundDescs)
    }

    num_workers = 16
    shuffle_indices = True

    wrapper = None
    wrapper_kwargs = {'load_spectrogram': False}
    postprocessor = None
    postprocessor_kwargs = {'num_tag_variants': 1,
                            'tag_token_max_length': 60,
                            'tag_model': 't5-small',
                            'token_max_length': 77,
                            'caption_model': 'gpt2', }

    output_root = '/home/t9s9/Datasets/beton/'
    output_name = 'audio_cap'
    out_path = Path(output_root) / output_name


@ex.named_config
def audio_only():
    output_name = 'CapGen_Audio'
    wrapper = AudioCapDatasetWrapper
    postprocessor = AudioCaptionPostprocessor


@ex.named_config
def opt_audio_only():
    output_name = 'CapGen_OPT_Audio'
    postprocessor_kwargs = dict(caption_model='facebook/opt-125m')
    wrapper = AudioCapDatasetWrapper
    postprocessor = AudioCaptionPostprocessor


@ex.named_config
def spectrogram():
    output_name = 'CapGen_Spectrogram'
    wrapper = AudioCapDatasetWrapper
    wrapper_kwargs = {'load_spectrogram': True}
    postprocessor = AudioSpectrogramCaptionPostprocessor
    postprocessor_kwargs = {'num_tag_variants': 1}


@ex.named_config
def audio_tag():
    output_name = 'CapGen_Audio+Tag_Roberta'
    wrapper = AudioCapDatasetWrapper
    postprocessor = AudioTagCaptionPostprocessor
    postprocessor_kwargs = {'num_tag_variants': 5,
                            'tag_token_max_length': 60,
                            'tag_model': 'roberta-base'}

@ex.named_config
def audio_tag_t5():
    output_name = 'CapGen_Audio+Tag_T5'
    wrapper = AudioCapDatasetWrapper
    postprocessor = AudioTagCaptionPostprocessor
    postprocessor_kwargs = {'num_tag_variants': 5,
                            'tag_token_max_length': 60,
                            'tag_model': 't5-base'}


@ex.automain
def main(_config):
    config = deepcopy(_config)
    train_ds, val_ds = create_datasets(config['wrapper'], config['datasets'].values(), config['wrapper_kwargs'])
    create_audio_beton(train_ds, val_ds,
                       postprocessor=config['postprocessor'],
                       output_path=config['out_path'],
                       postprocessor_kwargs=config['postprocessor_kwargs'],
                       num_workers=config['num_workers'],
                       shuffle_indices=config['shuffle_indices'])
