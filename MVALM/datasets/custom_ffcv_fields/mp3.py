from dataclasses import replace
from typing import Callable, Tuple, Type

import numpy as np
from fastmp3.libmp3 import mp3decode
from ffcv.fields.base import Field, ARG_TYPE
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State


class MP3Decoder(Operation):
    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return (
            replace(previous_state, jit_mode=True, shape=self.field.output_shape, dtype=self.field.output_dtype),
            AllocationQuery(self.field.output_shape, self.field.output_dtype)
        )

    def generate_code(self) -> Callable:
        mp3decode_c = Compiler.compile(mp3decode)

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                audio_data = mem_read(field, storage_state)

                size = mp3decode_c(audio_data, destination[dst_ix])
                destination[dst_ix, size:] = 0

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


MP3ArgsType = np.dtype([
    ('input_shape', '<u8', 32),  # 32 is the max number of dimensions for numpy
    ('output_shape', '<u8', 32),
])


class MP3Field(Field):
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_dtype = np.uint8()
        self.output_dtype = np.float32()

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<u8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        header_size = MP3ArgsType.itemsize
        header = binary[:header_size].view(MP3ArgsType)[0]
        input_shape = tuple(np.trim_zeros(header['input_shape'], trim='b'))
        output_shape = tuple(np.trim_zeros(header['output_shape'], trim='b'))
        return MP3Field(input_shape=input_shape, output_shape=output_shape)

    def to_binary(self) -> ARG_TYPE:
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        header = np.zeros(1, dtype=MP3ArgsType)
        s = np.array(self.input_shape).astype('<u8')
        header['input_shape'][0][:len(s)] = s
        s = np.array(self.output_shape).astype('<u8')
        header['output_shape'][0][:len(s)] = s
        to_write = header.view('<u1')
        result[0][:to_write.shape[0]] = to_write
        return result

    def encode(self, destination, field, malloc):
        destination[0], data_region = malloc(np.prod(self.input_shape) * self.input_dtype.itemsize)
        data_region[:] = field.reshape(-1).view('<u1')

    def get_decoder_class(self) -> Type[Operation]:
        return MP3Decoder
