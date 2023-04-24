from dataclasses import replace
from typing import Callable, Tuple, Type

import numpy as np
from PIL import Image
from ffcv.fields.base import Field, ARG_TYPE
from ffcv.fields.rgb_image import imdecode, IMAGE_MODES, resizer, encode_jpeg, memcpy
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State


class MultiRGBImageDecoder(Operation):
    """Most basic decoder for the :class:`~ffcv.fields.RGBImageField`.

    It only supports dataset with constant image resolution and will simply read (potentially decompress) and pass the images as is.
    """

    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']

        max_width = widths.max()
        max_height = heights.max()
        min_height = heights.min()
        min_width = widths.min()
        if min_width != max_width or max_height != min_height:
            msg = """SimpleRGBImageDecoder ony supports constant image,
consider RandomResizedCropRGBImageDecoder or CenterCropRGBImageDecoder
instead."""
            raise TypeError(msg)

        biggest_shape = (max_height, max_width, 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=biggest_shape, dtype=my_dtype),
            AllocationQuery(biggest_shape, my_dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        imdecode_c = Compiler.compile(imdecode)

        jpg = IMAGE_MODES['jpg']
        my_range = Compiler.get_iterator()
        my_memcpy = Compiler.compile(memcpy)

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]

                pointers = field['data_ptr']
                random_index = np.random.randint(0, len(pointers))
                image_data = mem_read(pointers[random_index], storage_state)
                height, width = field['height'], field['width']

                if field['mode'] == jpg:
                    imdecode_c(image_data, destination[dst_ix],
                               height, width, height, width, 0, 0, 1, 1, False, False)
                else:
                    my_memcpy(image_data, destination[dst_ix])

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


class MultiRGBImageField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting RGB image data.

    Parameters
    ----------
    write_mode : str, optional
        How to write the image data to the dataset file. Should be either 'raw'
        (``uint8`` pixel values), 'jpg' (compress to JPEG format), 'smart'
        (decide between saving pixel values and JPEG compressing based on image
        size), and 'proportion' (JPEG compress a random subset of the data with
        size specified by the ``compress_probability`` argument). By default: 'raw'.
    max_resolution : int, optional
        If specified, will resize images to have maximum side length equal to
        this value before saving, by default None
    jpeg_quality : int, optional
        The quality parameter for JPEG encoding (ignored for
        ``write_mode='raw'``), by default 90
    """

    def __init__(self,
                 num_images: int = 1,
                 write_mode='raw',
                 max_resolution: int = None,
                 jpeg_quality: int = 90) -> None:
        super().__init__()
        self.num_images = num_images
        self.write_mode = write_mode
        self.jpeg_quality = jpeg_quality
        self.max_resolution = max_resolution

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8', self.num_images),
        ])

    def get_decoder_class(self) -> Type[Operation]:
        return MultiRGBImageDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        size = np.dtype('<u8').itemsize
        num_images = binary[:size].view('<u8')
        num_images = num_images.item()
        return MultiRGBImageField(num_images=num_images)

    def to_binary(self) -> ARG_TYPE:
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        # add the information about the number of images to the binary
        s = np.array((self.num_images,)).astype('<u8')
        to_write = s.view('<u1')
        result[0][:len(to_write)] = to_write
        return result

    def encode(self, destination, images, malloc):
        def encode_singe_image(image):
            if self.write_mode == 'jpg':
                as_jpg = encode_jpeg(image, self.jpeg_quality)
                data_ptr, storage = malloc(as_jpg.nbytes)
                storage[:] = as_jpg
            elif self.write_mode == 'raw':
                image_bytes = np.ascontiguousarray(image).view('<u1').reshape(-1)
                data_ptr, storage = malloc(image.nbytes)
                storage[:] = image_bytes
            else:
                raise ValueError(f"Unsupported write mode {self.write_mode}")

            return data_ptr

        if not isinstance(images, (list, tuple)):
            raise ValueError(f"Images should be a list or tuple, got {type(images)}")

        if len(images) != self.num_images:
            raise ValueError(f"Expected {self.num_images} images, got {len(images)}")

        pointer = np.empty(self.num_images, dtype=np.uint64)
        for i, image in enumerate(images):
            if isinstance(image, Image.Image):
                image = np.array(image)

            if i == 0:
                destination['height'], destination['width'] = image.shape[:2]

            if not isinstance(image, np.ndarray):
                raise TypeError(f"Unsupported image type {type(image)}")

            if image.dtype != np.uint8:
                raise ValueError("Image type has to be uint8")

            if image.shape[2] != 3:
                raise ValueError(f"Invalid shape for rgb image: {image.shape}")

            assert image.dtype == np.uint8, "Image type has to be uint8"

            image = resizer(image, self.max_resolution)

            if destination['height'] != image.shape[0] or destination['width'] != image.shape[1]:
                raise ValueError(f"Images should have the same size, got {image.shape[:2]} and "
                                 f"{destination['height'], destination['width']}")

            pointer[i] = encode_singe_image(image)

        destination['mode'] = IMAGE_MODES[self.write_mode]
        destination['data_ptr'] = pointer
