from typing import Tuple, Union, Optional, List

import numpy as np
import packaging.version
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip
from torchvision.transforms.functional import to_pil_image

from .simple_tokenizer import SimpleTokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPTokenizer:
    def __init__(self, context_length: int = 77, truncate: bool = False):
        """

        :param context_length:  The context length to use; all CLIP models use 77 as the context length
        :param truncate: Whether to truncate the text in case its encoding is longer than the context length
        """
        self.tokenizer = SimpleTokenizer()
        self.context_length = context_length
        self.truncate = truncate
        self.vocab_size = len(self.tokenizer.encoder)

    def __call__(self, texts: Union[str, List[str]], prefix_special_token: Optional[str] = None) -> torch.Tensor:
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]

        if prefix_special_token is not None:
            if prefix_special_token not in self.tokenizer.cache.keys():
                raise ValueError(f"Unknown special token {prefix_special_token}")
            special_token = self.tokenizer.encoder[prefix_special_token]
            all_tokens = [[sot_token, special_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        else:
            all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]

        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                if self.truncate:
                    tokens = tokens[:self.context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input '{texts[i]}' is too long for context length {self.context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class CLIPImageProcessor:
    r"""
    Constructs a CLIP feature extractor.
    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 224):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        convert_rgb (`bool`, defaults to `True`):
            Whether or not to convert `PIL.Image.Image` into `RGB` format
    """

    def __init__(
            self,
            do_resize: bool = True,
            size: Union[int, Tuple[int, int]] = 224,
            resample: InterpolationMode = BICUBIC,
            do_center_crop: bool = True,
            crop_size: Union[int, Tuple[int, int]] = 224,
            do_normalize: bool = True,
            image_mean: Optional[Tuple[float, float, float]] = None,
            image_std: Optional[Tuple[float, float, float]] = None,
            do_convert_rgb: bool = True,
            random_flip: bool = False,
            return_tensors: bool = True,
    ):
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else CLIP_MEAN
        self.image_std = image_std if image_std is not None else CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.random_flip = random_flip
        self.return_tensors = return_tensors

        if do_normalize and not return_tensors:
            print("WARNING: do_normalize is set to True but return_tensors is False.")

        transforms = []
        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            transforms.append(_convert_image_to_rgb)
        if self.do_resize and self.size is not None and self.resample is not None:
            transforms.append(Resize(self.size, self.resample))
        if self.do_center_crop and self.crop_size is not None:
            transforms.append(CenterCrop(self.crop_size))
        if self.random_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        if self.do_normalize or return_tensors:
            transforms.append(ToTensor())
        if self.do_normalize:
            transforms.append(Normalize(self.image_mean, self.image_std))

        self.compose = Compose(transforms)

    def __call__(
            self,
            images: Union[
                Image.Image, np.ndarray, torch.Tensor, List[Image.Image], List[np.ndarray], List[torch.Tensor]
            ],
    ):
        """
        Main method to prepare for the model one or several image(s).

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:

        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray, torch.Tensor)):
                valid_images = True
        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or isinstance(images[0], torch.Tensor))
        )

        if not is_batched:
            images = [images]

        images = [to_pil_image(image) if not isinstance(image, Image.Image) else image for image in images]

        images = [self.compose(image) for image in images]

        if len(images) == 1:
            images = images[0]

        return images
