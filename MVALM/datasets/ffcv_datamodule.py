from typing import Union, List, Optional

import numpy as np
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, ToTensor, ToTorchImage
from pytorch_lightning import LightningDataModule

from MVALM.datasets.custom_ffcv_fields import MultiRGBImageField, MultiRGBImageDecoder, MP3Field
from MVALM.models.CLIP.processor import CLIP_MEAN, CLIP_STD
from MVALM.datasets.utils import to_3tuple


class FFCVDataModule(LightningDataModule):
    def __init__(self,
                 train_beton: str,
                 val_beton: Union[str, List[str]],
                 batch_size: Union[int, List[int]],
                 num_workers: Union[int, List[int]],
                 test_beton: Optional[Union[str, List[str]]] = None,
                 drop_last: Union[bool, List[bool]] = True,
                 os_cache: Union[bool, List[bool]] = True,
                 ordering: str = 'random',
                 batches_ahead: Union[int, List[int]] = 3,
                 seed: Optional[Union[int, List[int]]] = None,
                 distributed: bool = False,
                 image_mean: List[float] = CLIP_MEAN,
                 image_std: List[float] = CLIP_STD,
                 ):
        super().__init__()
        self.batch_size = to_3tuple(batch_size)
        self.num_workers = to_3tuple(num_workers)
        self.drop_last = to_3tuple(drop_last)
        self.os_cache = to_3tuple(os_cache)
        self.distributed = distributed
        self.batches_ahead = to_3tuple(batches_ahead)
        self.seed = to_3tuple(seed)

        order_option = dict(random=OrderOption.RANDOM,
                            sequential=OrderOption.SEQUENTIAL,
                            quasi_random=OrderOption.QUASI_RANDOM)

        assert ordering in order_option.keys(), f'Ordering {ordering} not supported. Choose from {order_option.keys()}'
        self.ordering = order_option[ordering]
        self.train_pipelines = {}
        self.test_pipelines = {}
        self.custom_fields_train = {}
        self.custom_fields_test = {}

        self.train_beton = train_beton
        self.val_beton = [val_beton] if isinstance(val_beton, str) else val_beton
        self.test_beton = [test_beton] if isinstance(test_beton, str) else test_beton

        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]

    def setup(self, stage: Optional[str] = None) -> None:
        train_image_transform = [
            MultiRGBImageDecoder(),
            RandomHorizontalFlip(flip_prob=0.5),
            NormalizeImage(mean=np.array(self.image_mean) * 255,
                           std=np.array(self.image_std) * 255,
                           type=np.dtype('float32')),
            ToTensor(),
            ToTorchImage(),
        ]
        test_image_transform = [
            SimpleRGBImageDecoder(),
            NormalizeImage(mean=np.array(self.image_mean) * 255,
                           std=np.array(self.image_std) * 255,
                           type=np.dtype('float32')),
            ToTensor(),
            ToTorchImage(),
        ]

        self.train_pipelines = {'image': train_image_transform}
        self.test_pipelines = {'image': test_image_transform}

        self.custom_fields_train = {'audio': MP3Field, 'image': MultiRGBImageField}
        self.custom_fields_test = {'audio': MP3Field}

    def train_dataloader(self) -> Loader:
        train_args = dict(batch_size=self.batch_size[0],
                          num_workers=self.num_workers[0],
                          os_cache=self.os_cache[0],
                          distributed=self.distributed,
                          seed=self.seed[0],
                          drop_last=self.drop_last[0],
                          batches_ahead=self.batches_ahead[0],
                          order=self.ordering,
                          pipelines=self.train_pipelines,
                          custom_fields=self.custom_fields_train)
        return Loader(self.train_beton, **train_args)

    def val_dataloader(self) -> List[Loader]:
        val_args = dict(batch_size=self.batch_size[1],
                        num_workers=self.num_workers[1],
                        os_cache=self.os_cache[1],
                        distributed=self.distributed,
                        seed=self.seed[1],
                        drop_last=self.drop_last[1],
                        batches_ahead=self.batches_ahead[1],
                        order=OrderOption.SEQUENTIAL,
                        pipelines=self.test_pipelines,
                        custom_fields=self.custom_fields_test)

        return [Loader(val_beton, **val_args) for val_beton in self.val_beton]

    def test_dataloader(self) -> List[Loader]:
        test_args = dict(batch_size=self.batch_size[2],
                         num_workers=self.num_workers[2],
                         os_cache=self.os_cache[2],
                         distributed=self.distributed,
                         seed=self.seed[2],
                         drop_last=self.drop_last[2],
                         batches_ahead=self.batches_ahead[2],
                         order=OrderOption.SEQUENTIAL,
                         pipelines=self.test_pipelines,
                         custom_fields=self.custom_fields_test)

        return [Loader(test_beton, **test_args) for test_beton in self.test_beton]
