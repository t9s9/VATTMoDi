import json
from dataclasses import replace
from typing import Callable, Tuple, Type

import numpy as np
from ffcv.fields.base import Field, ARG_TYPE
from ffcv.libffcv import memcpy
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

FlexNDArrayArgsType = np.dtype([
    ('fixed_dim', '<u8'),  # size of the last and fixed dimension
    ('type_length', '<u8'),  # length of the dtype description
])


class FlexNDArrayDecoder(Operation):
    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return (
            replace(previous_state, jit_mode=True, shape=(self.field.fixed_dim,), dtype=self.field.dtype),
            AllocationQuery((self.field.fixed_dim,), self.field.dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        my_memcpy = Compiler.compile(memcpy)

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]

                array_data = mem_read(field['data_ptr'], storage_state)

                nr_samples = field['samples']
                rand_ix = np.random.randint(0, nr_samples)
                array_data = array_data.reshape(nr_samples, -1)[rand_ix]

                my_memcpy(array_data, destination[dst_ix])
            return destination[:len(batch_indices)]

        # decode.is_parallel = True
        return decode


class FlexNDArrayField(Field):
    def __init__(self, dtype: np.dtype, fixed_dim: int) -> None:
        self.dtype = dtype
        self.fixed_dim = fixed_dim

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('samples', '<u8'),
            ('data_ptr', '<u8'),
        ])

    def get_decoder_class(self) -> Type[Operation]:
        return FlexNDArrayDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        header_size = FlexNDArrayArgsType.itemsize
        header = binary[:header_size].view(FlexNDArrayArgsType)[0]
        type_length = header['type_length']
        type_data = binary[header_size:][:type_length].tobytes().decode('ascii')
        type_desc = json.loads(type_data)
        type_desc = [tuple(x) for x in type_desc]
        assert len(type_desc) == 1
        dtype = np.dtype(type_desc)['f0']
        fixed_dim = header['fixed_dim']
        return FlexNDArrayField(dtype, fixed_dim)

    def to_binary(self) -> ARG_TYPE:
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        header = np.zeros(1, dtype=FlexNDArrayArgsType)
        header['fixed_dim'] = self.fixed_dim
        encoded_type = json.dumps(self.dtype.descr)
        encoded_type = np.frombuffer(encoded_type.encode('ascii'), dtype='<u1')
        header['type_length'] = len(encoded_type)
        to_write = np.concatenate([header.view('<u1'), encoded_type])
        result[0][:to_write.shape[0]] = to_write
        return result

    def encode(self, destination, array, malloc):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Unsupported array type {type(array)}")

        if not array.ndim == 2:
            raise ValueError(f"Unsupported number of dimensions {array.shape}")

        if not array.shape[1] == self.fixed_dim:
            raise ValueError(f"Unsupported number of features {array.shape[1]} but {self.fixed_dim} given")

        if not array.dtype == self.dtype:
            raise ValueError(f"Unsupported dtype {array.dtype} but {self.dtype} given")

        # first dim of shape is number of samples
        destination['samples'] = array.shape[0]

        array_bytes = array.view('<u1').reshape(-1)
        destination['data_ptr'], storage = malloc(array.nbytes)
        storage[:] = array_bytes
