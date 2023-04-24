from typing import List

import pytorch_lightning.callbacks as clb


def get_callbacks(config) -> List[clb.Callback]:
    callbacks = [
        clb.RichModelSummary(max_depth=3 if config['debug'] else 1),
    ]

    if not config['debug']:
        callbacks.append(clb.LearningRateMonitor(logging_interval="step", log_momentum=False))

        if config['callbacks']['early_stopping']['enabled']:
            # monitor='val_loss', patience=2, mode='min'
            callbacks.append(clb.EarlyStopping(**config['callbacks']['early_stopping']['params']))
        if config['callbacks']['model_checkpoint']['enabled']:
            # save_last=False, save_top_k=3, monitor="val_loss", mode='min', verbose=False,
            # save_weights_only=False, filename=f"{config['exp_name']}" + "-{epoch}-{val_loss:.3f}"
            ckpt = clb.ModelCheckpoint(
                dirpath=config['checkpoint_path'],
                **config['callbacks']['model_checkpoint']['params']
            )
            ckpt.CHECKPOINT_NAME_LAST = f"{config['exp_name']}-last"
            callbacks.append(ckpt)

    if config['enable_progress_bar']:
        callbacks.append(clb.RichProgressBar(refresh_rate=1))

    return callbacks
