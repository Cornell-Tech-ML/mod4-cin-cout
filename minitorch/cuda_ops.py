# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, Dict

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


# change to a cuda function
def device_jit(fn: Fn, **kwargs: Dict[str, Any]) -> Fn:
    """Convert a Python function to a CUDA device function.

    Parameters
    ----------
    fn : Fn
        The Python function to compile for CUDA device use.
    **kwargs : dict
        Additional arguments for the CUDA JIT compiler.

    Returns
    -------
    Fn
        The compiled CUDA device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Dict[str, Any]) -> FakeCUDAKernel:
    """Compile a Python function into a CUDA kernel.

    Parameters
    ----------
    fn : Callable
        The Python function to compile for CUDA kernel execution.
    **kwargs : dict
        Additional arguments for the CUDA JIT compiler.

    Returns
    -------
    FakeCUDAKernel
        The compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            # decide how many block in a grid
            # + THREADS_PER_BLOCK - 1 cuz we need to round up to the nearest integer
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            # 1D block with 1D thread
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            # it's for cuda reduce so out shape will be block size. It's more efficent for cuda to do reduce
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # to store the tmp index
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # to store the tmp index
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        # cuda.blockDim.x = threadperblock
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # Guard
        if i < out_size:
            # change i to the outindex, for example 0 -> (0,0,0)
            to_index(i, out_shape, out_index)
            # reverse the broadcast index to orginal index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            # map the input position to out put
            out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # Guard
        if i < out_size:
            # change i to the outindex, for example 0 -> (0,0,0)
            to_index(i, out_shape, out_index)
            # reverse the broadcast index to corresponding index a
            broadcast_index(out_index, out_shape, a_shape, a_index)
            # reverse the broadcast index to corresponding index in b
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out[i] = fn(
                # get the position in a and b from their index
                a_storage[index_to_position(a_index, a_strides)],
                b_storage[index_to_position(b_index, b_strides)],
            )

    # compile low level function to cuda kernel function
    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length n and out of size n // blockDIM
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # evert block has its own share memeory.
    # for example, a = 0~64. 0~31 will be put into block0's cache and 32~64 will be put into block1' cache
    cache[pos] = a[i] if i < size else 0.0
    # wait till every thread in "a block" finish assignment
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        # for sum the cache in every block
        # for example
        # first loop : [1, 2, 3, 4, 5, 6, 7, 8]
        # second loop : [1+2, 2, 3+4, 4, 5+6, 6, 7+8, 8]
        # last loop : [10+26, 2, 7, 4, 26, 6, 15, 8]
        # cache[0] will be the sum of the first block
        if pos % (2 * stride) == 0:
            cache[pos] += cache[pos + stride]
        stride *= 2
        # wait every thread in a block finish their first loop
        cuda.syncthreads()

    # Write result from each block to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the block-wise sum of elements in the input tensor.

    This function utilizes a CUDA kernel to sum elements of the input tensor `a`
    block-by-block. The results of each block's sum are stored in the output tensor `out`.

    Parameters
    ----------
    a : Tensor
        The input tensor containing the elements to be summed.

    Returns
    -------
    TensorData
        A tensor containing the sums of each block's elements.
        The output tensor has a fixed size of 2 for simplicity, with results
        computed using CUDA.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # blocks represnt the position in output tensor
        # threads deal with each value in "dim" in a (input tenosr)

        # how many thread in a block
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        if out_pos < out_size:
            # compute out_index from out_pos
            to_index(out_pos, out_shape, out_index)

            # get the block index along reduce_dim
            reduce_block_idx = out_index[reduce_dim]

            # compute reduce_start and reduce_end
            reduce_start = reduce_block_idx * BLOCK_DIM
            reduce_end = min(reduce_start + BLOCK_DIM, a_shape[reduce_dim])
            # compute the number of active threads for this block
            active_threads = reduce_end - reduce_start

            # a_index will be same as out index, except for reduce dim
            for d in range(len(a_shape)):
                a_index[d] = out_index[d]

            # defualt value will be different depend on mul reduce or add reduce
            acc = reduce_value

            # active thread
            if pos < active_threads:
                # idx = the position in reduce dim this thread dealing w
                idx = reduce_start + pos
                a_index[reduce_dim] = idx

                # get the value of this index
                a_pos = index_to_position(a_index, a_strides)
                acc = a_storage[a_pos]

            else:
                # For threads beyond active_threads, set acc to default value
                acc = reduce_value

            # sharing cache store the result of each thread in current block
            # ex reduce dim 1 (0,0) (0,1) ... (0, 1024) will store in cache [0] [1] [2]
            cache[pos] = acc
            # wait every thread
            cuda.syncthreads()

            # Perform reduction in shared memory, using binary reduction
            stride = 1
            while stride < BLOCK_DIM:
                # for sum the cache in every block
                # for example
                # first loop : [1, 2, 3, 4, 5, 6, 7, 8]
                # second loop : [1+2, 2, 3+4, 4, 5+6, 6, 7+8, 8]
                # last loop : [10+26, 2, 7, 4, 26, 6, 15, 8]
                # cache[0] will be the sum of the first block = the value after reduction
                index = 2 * stride * pos
                if index + stride < BLOCK_DIM and (index + stride) < active_threads:
                    cache[index] = fn(cache[index], cache[index + stride])
                cuda.syncthreads()
                stride *= 2

            # store the value in out position when threadid = 0 in each block
            if pos == 0:
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block (local to the block).
    # Each thread works on a specific row and column within the block.
    local_i = cuda.threadIdx.y  # Row within the block
    local_j = cuda.threadIdx.x  # Column within the block

    # Compute the global indices (row and column) in the output matrix `out`
    # corresponding to the current thread in the block.
    i = cuda.blockIdx.x * cuda.blockDim.x + local_i
    j = cuda.blockIdx.y * cuda.blockDim.y + local_j

    # Initialize an accumulator for the dot product.
    # This will store the partial sum of the dot product for this thread's output element.
    acc = 0.0

    # Loop over the shared dimension of the matrices (number of columns in A = number of rows in B).
    # Each thread loads one element of the tile.
    # The matrices are divided into tiles of size BLOCK_DIM, and the loop processes one tile at a time.
    for k in range(0, size, BLOCK_DIM):
        # Load tiles of A and B into shared memory
        if i < size and (k + local_j) < size:
            # strides [size, 1] so a[i * size + (k + local_j)] = a[i, local_j]
            a_shared[local_i, local_j] = a[i * size + (k + local_j)]
        else:
            a_shared[local_i, local_j] = 0.0

        if j < size and (k + local_i) < size:
            # strides [size, 1] so b[(k + local_i) * size + j] = b[local_i, j]
            b_shared[local_i, local_j] = b[(k + local_i) * size + j]
        else:
            b_shared[local_i, local_j] = 0.0

        # Synchronize all threads in the block to ensure the tiles are fully loaded into shared memory.
        cuda.syncthreads()

        # Perform the computation for the current tile.
        # Each thread computes a partial sum of the dot product for its corresponding output element.
        for local_k in range(min(size - k, BLOCK_DIM)):
            acc += a_shared[local_i, local_k] * b_shared[local_k, local_j]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Write the computed value to global memory
    # Guard
    if i < size and j < size:
        # strides [size, 1] so out[i * size + j] = out[i, j]
        out[i * size + j] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication for small square matrices using shared memory.

    This function computes the matrix product of two input tensors `a` and `b`,
    assuming they are square matrices of the same size. The computation is optimized
    by using shared memory for intermediate results, minimizing global memory accesses.

    Parameters
    ----------
    a : Tensor
        The first input tensor (matrix).
    b : Tensor
        The second input tensor (matrix).

    Returns
    -------
    TensorData
        A tensor containing the result of the matrix multiplication.
        The output tensor has the same dimensions as the input matrices.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    assert a_shape[-1] == b_shape[-2]
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    # refer to the image in GPU puzzle
    # This CUDA kernel implements matrix multiplication using shared memory for efficient computation.
    # The code divides the input matrices `a` and `b` into tiled blocks.
    # Each tile is collaboratively loaded into shared memory by threads to minimize global memory access.

    # Each thread computes a single output element in the result matrix `out`.
    # It does so by iterating over shared memory chunks and accumulating the dot product
    # for the corresponding row of `a` and column of `b`.

    # To handle edge cases where the matrix dimensions are not divisible by the block size,
    # the code uses zero-padding for out-of-bounds tiles. This ensures correctness
    # and prevents invalid memory access.

    # Synchronization points (cuda.syncthreads) are strategically placed to ensure
    # that all threads finish loading shared memory before proceeding to computation.

    # By leveraging shared memory to reduce access latency, this approach optimizes memory bandwidth usage
    # and achieves high parallel efficiency, making it well-suited for large-scale matrix operations.

    # each thread per block will have its own acc
    acc = 0.0

    # Iterate over the shared dimension (number of columns in `a` = rows in `b`).
    for k in range(0, a_shape[-1], BLOCK_DIM):
        # Guard
        if i < a_shape[-2] and k + pj < a_shape[-1]:
            # Each thread loads one element of `a` into shared memory.
            # this mean a_share = a[i, k + local_i]
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[-2] + (k + pj) * a_strides[-1]
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Guard
        if j < b_shape[-1] and k + pi < b_shape[-2]:
            # this mean b_share = a[i, k + local_i]
            # Each thread loads one element of `b` into shared memory.
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + (k + pi) * b_strides[-2] + j * b_strides[-1]
            ]
        else:
            b_shared[pi, pj] = 0.0

        cuda.syncthreads()

        # Compute the partial dot product for the block.
        # For example, when pi=0 and pj=0, the thread will compute the sum of products: a_shared[0, 0] * b_shared[0, 0] + a_shared[0, 1] * b_shared[1, 0] + ... + a_shared[0, TPB-1] * b_shared[TPB-1, 0].
        for local_k in range(min(BLOCK_DIM, a_shape[-1] - k)):
            acc += a_shared[pi, local_k] * b_shared[local_k, pj]
        cuda.syncthreads()

    # Guard
    if i < out_shape[-2] and j < out_shape[-1]:
        # use batch i j (index) with stride to get the out position
        out_pos = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        # Store the accumulated result in the output matrix.
        out[out_pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
