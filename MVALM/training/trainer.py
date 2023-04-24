from logging import Logger

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from MVALM.datasets.ffcv_datamodule import FFCVDataModule
from MVALM.modules.vat_module import VATModule, ATModule
from MVALM.training import prepare_workspace, finalize_workspace, get_callbacks, get_logger
from MVALM.evaluation.zero_shot import ZeroShotCallback


def train(_config, _log: Logger):
    _log.info("Prepare Workspace")
    config = prepare_workspace(_config)

    _log.info("Initialize Model")
    module = VATModule(loss_kwargs=config['loss'],
                       optimizer_kwargs=config['optimizer'],
                       lr_scheduler_kwargs=config['lr_scheduler'],
                       caption_type=config['caption_type'],
                       model_kwargs=config['model'],
                       sync_grads=config['sync_grads'],
                       gather_features=config['gather_features'])
    if hasattr(module.model, 'vision_encoder'):
        processing_kwargs = module.model.vision_encoder.preprocessor_kwargs()
        mean, std = processing_kwargs['mean'], processing_kwargs['std']
    else:
        mean, std = None, None

    _log.info("Initialize Datamodule")
    datamodule = FFCVDataModule(**config['data'],
                                image_mean=mean, image_std=std)

    _log.info("Initialize Logger")
    logger = get_logger(config)

    _log.info("Initialize Trainer")
    callbacks = get_callbacks(config)

    for zs in config['zero_shot']:
        callbacks.append(ZeroShotCallback(**zs,
                                          tokenizer=module.model.text_encoder.tokenize,
                                          modality_forward=lambda x, vat_module: vat_module(spectrogram=x).a_proj,
                                          text_forward=lambda x, vat_module: vat_module(input_ids=x.input_ids,
                                                                                        attention_mask=x.attention_mask).t_proj,

                                          )
                         )

    trainer = pl.Trainer(
        num_sanity_val_steps=4,
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        precision=config['precision'],
        devices=len(config['gpus']),
        strategy=DDPStrategy(static_graph=True) if len(config['gpus']) > 1 else 'auto',
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
        sync_batchnorm=config['sync_batchnorm'],
        gradient_clip_val=config['gradient_clip_val'],
        gradient_clip_algorithm=config['gradient_clip_algorithm'],
    )

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=config['resume_from_checkpoint'])

    if not config['debug'] and config['do_test']:
        trainer.test(datamodule=datamodule, model=module, ckpt_path='best')

    finalize_workspace()
