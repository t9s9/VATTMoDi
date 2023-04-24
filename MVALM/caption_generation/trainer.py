from logging import Logger

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from MVALM.datasets import AudioCapsAudioOnly, Clotho
from MVALM.datasets.FeatureDataset.dataloader import FeatureDataloader
from MVALM.training import prepare_workspace, finalize_workspace, get_callbacks, get_logger
from dataset import AudioCapDatasetWrapper
from evaluation_callback import CaptionGenerationCallback
from module import CaptioningModule


def train(_config, _log: Logger):
    _log.info("Prepare Workspace")
    config = prepare_workspace(_config)

    _log.info("Initialize Model")
    module = CaptioningModule(
        lr=config['lr'],
        num_warmup_steps=config['num_warmup_steps'],
        train_gpt=config['train_gpt'],
        **config['model']
    )

    _log.info("Initialize Datamodule")
    datamodule = FeatureDataloader(**config['data'])

    _log.info("Initialize Logger")
    logger = get_logger(config)

    _log.info("Initialize Trainer")
    callbacks = get_callbacks(config)

    dataset = AudioCapsAudioOnly(datasets_root=config['test_root']['AudioCaps'], split='test')
    # callbacks.append(CaptionGenerationCallback(dataset, perform_on_validation=True, perform_on_test=False, limit=None))

    trainer = pl.Trainer(
        num_sanity_val_steps=4,
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        precision=config['precision'],
        devices=len(config['gpus']),
        strategy="auto",
        accelerator='gpu',
        num_nodes=1,
        detect_anomaly=False,
        fast_dev_run=config['debug'],
        enable_progress_bar=config['enable_progress_bar'],
        val_check_interval=config['val_check_interval'],
        use_distributed_sampler=False,  # to False if FFCV
        enable_model_summary=False,
        overfit_batches=config['overfit_batches'],
    )

    trainer.fit(model=module, datamodule=datamodule)

    if not config['debug'] and config['do_test']:
        datasets = [AudioCapsAudioOnly(datasets_root=config['test_root']['AudioCaps'], split='test'),
                    Clotho(datasets_root=config['test_root']['Clotho'], split='test')]

        for dataset in datasets:
            trainer.test(
                dataloaders=DataLoader(
                    AudioCapDatasetWrapper(dataset, choose_caption=False),
                    num_workers=8, batch_size=None),
                model=module, verbose=True, ckpt_path='best'
            )

    finalize_workspace()
