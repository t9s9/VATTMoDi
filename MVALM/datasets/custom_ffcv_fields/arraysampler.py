from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
import numpy as np


class ArraySampler(Operation):
    """Sample along a given dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def generate_code(self) -> Callable:
        dim = self.dim
        my_range = Compiler.get_iterator()

        def sample(inp, dst):
            nr_samples = inp.shape[dim + 1]

            for batch_idx in my_range(inp.shape[0]):
                rand_ix = np.random.randint(0, nr_samples)
                inp[batch_idx].take(rand_ix, dim, out=dst[batch_idx])
            return dst

        return sample

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = [s for i, s in enumerate(previous_state.shape) if i != self.dim]
        return (
            replace(previous_state, shape=new_shape, jit_mode=False),
            AllocationQuery(shape=new_shape, dtype=previous_state.dtype)
        )
