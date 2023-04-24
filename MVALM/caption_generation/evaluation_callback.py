from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from dataset import AudioCapDatasetWrapper
from evaluation import score_auditory_captions


class CaptionGenerationCallback(pl.Callback):
    def __init__(self,
                 dataset,
                 perform_on_validation: bool = True,
                 perform_on_test: bool = False,
                 log_samples: int = 5,
                 num_workers: int = 2,
                 limit: Optional[int] = None,
                 ):
        super().__init__()
        self.dataset_name = getattr(dataset, 'name', 'validation_dataset')

        dataset = AudioCapDatasetWrapper(dataset, choose_caption=False)
        if limit is not None:
            dataset = Subset(dataset, range(limit))
        self.dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)

        self.perform_on_validation = perform_on_validation
        self.perform_on_test = perform_on_test
        self.log_samples = log_samples

        self.generations = []

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.perform_on_validation:
            self.run(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.perform_on_test:
            self.run(trainer, pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(trainer.logger, 'log_text'):
            trainer.logger.log_text(key=f"samples", dataframe=pd.DataFrame(self.generations))

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # if trainer.is_global_zero and not trainer.sanity_checking and not trainer.fast_dev_run:
        if True:
            print(f"Running Evaluation on {self.dataset_name}.")
            torch.cuda.empty_cache()
            pl_module.eval()

            outputs = []
            meta = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for i, batch in enumerate(self.dataloader):
                    batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
                    outputs.append(pl_module.test_step(batch, i))

                    if i < self.log_samples:
                        meta.append(
                            dict(filename=batch.filename, caption=list(batch.caption), tags=list(batch.tags),
                                 step=trainer.global_step))

            preds, target = zip(*outputs)
            scores = score_auditory_captions(preds=preds, gts=target, metric_prefix=self.dataset_name,
                                             bleu=pl_module.test_kwargs['bleu'], rouge=pl_module.test_kwargs['rouge'],
                                             cider=pl_module.test_kwargs['cider'], spice=pl_module.test_kwargs['spice'],
                                             meteor=pl_module.test_kwargs['meteor'],
                                             fense=pl_module.test_kwargs['fense'])

            for i, m in enumerate(meta):
                self.generations.append(dict(prediction=preds[i], **m))

            for metric, value in scores.items():
                print(f"{metric}: {value:.3f}")
                if hasattr(trainer.logger, 'log_metrics'):
                    trainer.logger.log_metrics({f'{metric}': value}, step=trainer.global_step)

            pl_module.train()
            torch.cuda.empty_cache()
