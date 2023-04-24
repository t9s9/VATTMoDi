import gc
from typing import List, Tuple, Union, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .zero_shot_data import get_classes, get_templates
from .zero_shot_func import _create_zero_shot_classifier, _evaluate_zero_shot
from ...models import CLIPTokenizer


class ZeroShotCallback(pl.Callback):
    def __init__(self,
                 dataset: Union[str, Dataset],
                 modality_forward=None,
                 text_forward=None,
                 dataset_name: str = None,
                 batch_size: int = 64,
                 num_workers: int = 2,
                 simple_templates: bool = False,
                 top_k: Tuple[int, ...] = (1, 2, 5, 10),
                 accuracy_average: str = 'micro',
                 verbose: bool = False,
                 classes: List = None,
                 templates: List = None,
                 tokenizer: CLIPTokenizer = None,
                 prefix_special_token: Optional[str] = None,
                 perform_on_validation: bool = True,
                 perform_on_test: bool = False):
        super().__init__()
        self.dataset = dataset
        if isinstance(self.dataset, str):
            from ffcv.loader import Loader, OrderOption
            self.dataloader = Loader(self.dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     os_cache=True,
                                     distributed=False,
                                     drop_last=False,
                                     order=OrderOption.SEQUENTIAL)
        else:
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=False)
        self.modality_forward = modality_forward
        self.text_forward = text_forward
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.accuracy_average = accuracy_average
        self.simple_templates = simple_templates
        self.verbose = verbose
        self.prefix_special_token = prefix_special_token
        self.perform_on_validation = perform_on_validation
        self.tokenizer = CLIPTokenizer(context_length=77, truncate=True) if tokenizer is None else tokenizer
        if self.prefix_special_token is not None:
            self.tokenizer.tokenizer.add_special_tokens(['<|visual|>', '<|auditory|>'])

        self.perform_on_test = perform_on_test
        self.classes = get_classes(dataset_name) if classes is None else classes
        self.templates = get_templates(dataset_name, simple=self.simple_templates) if templates is None else templates

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.perform_on_validation:
            self.run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.perform_on_test:
            self.run(trainer, pl_module)

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.is_global_zero and not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            pl_module.model.eval()
            device = pl_module.device
            if self.verbose:
                print("Performing Zero-Shot-Evaluation on dataset: ", self.dataset_name, device)

            classifier = _create_zero_shot_classifier(forward_func=lambda i: self.text_forward(i, pl_module),
                                                      classnames=self.classes,
                                                      templates=self.templates,
                                                      tokenizer=self.tokenizer,
                                                      batch_size=self.batch_size,
                                                      device=device,
                                                      prefix_special_token=self.prefix_special_token,
                                                      verbose=self.verbose)

            result = _evaluate_zero_shot(forward_func=lambda i: self.modality_forward(i, pl_module),
                                         classifier=classifier,
                                         dataloader=self.dataloader,
                                         top_k=self.top_k,
                                         average=self.accuracy_average,
                                         device=device,
                                         dtype=torch.float16,
                                         verbose=self.verbose)

            for k, value in result.items():
                if hasattr(trainer.logger, 'log_metrics'):
                    trainer.logger.log_metrics({f'zs/{self.dataset_name}_{k}': value}, step=trainer.global_step)

            del classifier
            gc.collect()
            torch.cuda.empty_cache()
            pl_module.model.train()
