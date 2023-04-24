from logging import Logger

import pytorch_lightning as pl
import torch

from MVALM.datasets.ffcv_datamodule import FFCVDataModule
from MVALM.evaluation.classification.classifier import Classifier
from MVALM.models.encoder import AudioEncoder
from MVALM.training import prepare_workspace, finalize_workspace, get_callbacks, get_logger
from MVALM.evaluation.classification.utils import load_audio_model


def train(_config, _log: Logger):
    _log.info("Prepare Workspace")
    config = prepare_workspace(_config)

    _log.info("Initialize Model")
    model: AudioEncoder = load_audio_model(**config['model'], freeze=config['freeze_backbone'])

    module = Classifier(model=model,
                        task=config['task'],
                        dropout=config['dropout'],
                        num_classes=config['num_classes'],
                        freeze_backbone=config['freeze_backbone'],
                        lr=config['lr'],
                        weight_decay=config['weight_decay'],
                        warmup_epochs=config['warmup_epochs'])

    if hasattr(torch, "compile"):
        _log.info("Compile Model")
        module = torch.compile(module, mode='max-autotune')

    _log.info("Initialize Datamodule")
    datamodule = FFCVDataModule(**config['data'])

    _log.info("Initialize Logger")
    logger = get_logger(config)

    _log.info("Initialize Trainer")
    callbacks = get_callbacks(config)

    trainer = pl.Trainer(
        num_sanity_val_steps=4,
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        precision=config['precision'],
        devices=len(config['gpus']),
        strategy='auto',
        accelerator='gpu',
        num_nodes=1,
        detect_anomaly=False,
        fast_dev_run=config['debug'],
        enable_progress_bar=config['enable_progress_bar'],
        val_check_interval=config['val_check_interval'],
        use_distributed_sampler=False,  # to False if FFCV
        enable_model_summary=False,
        overfit_batches=config['overfit_batches'],
        enable_checkpointing=not config['debug'],
    )

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=config['resume_from_checkpoint'])

    if not config['debug'] and config['do_test']:
        trainer.test(datamodule=datamodule, model=module, ckpt_path='best')

    finalize_workspace()
