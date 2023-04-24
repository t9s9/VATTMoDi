from typing import Union, List, Optional

import numpy as np
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, ToTensor, ToTorchImage, ToDevice
from pytorch_lightning import LightningDataModule

from MVALM.datasets.custom_ffcv_fields import FlexNDArrayField
from MVALM.models.CLIP.processor import CLIP_MEAN, CLIP_STD
from MVALM.datasets.utils import to_3tuple


class FeatureDataloader(LightningDataModule):
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

        assert ordering in order_option.keys()
        self.ordering = order_option[ordering]
        self.train_pipelines = {}
        self.test_pipelines = {}
        self.custom_fields_train = {}
        self.custom_fields_test = {}

        self.train_beton = train_beton
        self.val_beton = [val_beton] if isinstance(val_beton, str) else val_beton
        self.test_beton = [test_beton] if isinstance(test_beton, str) else test_beton

    # Subclass and override this method
    # def setup(self, stage: Optional[str] = None) -> None:
    #     train_image_transform = [SimpleRGBImageDecoder(),
    #                              RandomHorizontalFlip(flip_prob=0.5),
    #                              NormalizeImage(mean=np.array(CLIP_MEAN) * 255,
    #                                             std=np.array(CLIP_STD) * 255,
    #                                             type=np.dtype('float32')),
    #                              ToTensor(),
    #                              ToTorchImage(),
    #                              # ToDevice(torch.device('cuda'), non_blocking=True),
    #                              ]
    #     test_image_transform = [SimpleRGBImageDecoder(),
    #                             NormalizeImage(mean=np.array(CLIP_MEAN) * 255,
    #                                            std=np.array(CLIP_STD) * 255,
    #                                            type=np.float32),
    #                             ToTensor(),
    #                             ToTorchImage(),
    #                             ]
    #
    #     self.train_pipelines = {f'image_{i}': train_image_transform for i in range(3)}
    #     self.test_pipelines = {'image_0': test_image_transform}
    #     self.custom_fields_train = {'visual_caption': FlexNDArrayField, 'auditory_caption': FlexNDArrayField}
    #     self.custom_fields_test = {}

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
