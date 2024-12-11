from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    input = input.contiguous()

    new_height = height // kh
    new_width = width // kw
    out = input.view(batch, channel, new_height, kh, new_width, kw)
    out = out.permute(0, 1, 2, 4, 3, 5)
    out = out.contiguous()
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling on the 2D input tensor.

    Parameters
    ----------
    input : Tensor
        Input tensor with shape [batch, channels, height, width].
    kernel : Tuple[int, int]
        Tuple specifying the height and width of the pooling kernel.

    Returns
    -------
    Tensor
        Tensor after applying average pooling, with reduced height and width
        depending on the kernel size.
        Shape: [batch, channels, new_height, new_width]

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum values along a specified dimension.

        Parameters
        ----------
        ctx : Context
            Context object for storing intermediate values for the backward pass.
        t : Tensor
            Input tensor.
        dim : Tensor
            Dimension along which to compute the maximum.

        Returns
        -------
        Tensor
            A tensor containing the maximum values along the specified dimension.

        """
        d = int(dim.item())
        res = FastOps.reduce(operators.max, start=-1e30)(t, d)
        ctx.save_for_backward(t, dim, res)
        return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the max operation.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        grad_output : Tensor
            Gradient of the loss with respect to the output of the max operation.

        Returns
        -------
        Tuple[Tensor, float]
            - Gradient of the loss with respect to the input tensor.
            - A float representing the gradient with respect to the dimension, which is always 0.

        """
        t, dim, max_val = ctx.saved_values
        d = int(dim.item())
        mask = t == max_val
        sum_mask = mask.sum(dim=d)
        grad_input = mask * (grad_output / sum_mask)
        return grad_input, 0.0


def max(t: Tensor, dim: int) -> Tensor:
    """Apply the max function along a specified dimension.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    dim : int
        Dimension along which to compute the maximum.

    Returns
    -------
    Tensor
        Tensor containing the maximum values along the specified dimension.

    """
    return Max.apply(t, tensor(dim))


def argmax(t: Tensor, dim: int) -> Tensor:
    """Compute the indices of the maximum values along a specified dimension.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    dim : int
        Dimension along which to compute the indices of the maximum.

    Returns
    -------
    Tensor
        Tensor containing one-hot encoded indices of the maximum values along the specified dimension.

    """
    m = max(t, dim)
    expand_shape = list(m.shape)
    expand_shape.insert(dim, t.shape[dim])
    mask = t == m
    return mask


def softmax(t: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specified dimension.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    dim : int
        Dimension along which to compute the softmax.

    Returns
    -------
    Tensor
        Tensor containing the softmax probabilities along the specified dimension.

    """
    exp_t = t.exp()
    sum_exp = exp_t.sum(dim=dim)
    return exp_t / sum_exp


def logsoftmax(t: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along a specified dimension.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    dim : int
        Dimension along which to compute the logsoftmax.

    Returns
    -------
    Tensor
        Tensor containing the log of the softmax probabilities along the specified dimension.

    """
    m = max(t, dim=dim)
    log_sum_exp = ((t - m).exp().sum(dim=dim)).log() + m
    return t - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling on the 2D input tensor.

    Parameters
    ----------
    input : Tensor
        Input tensor with shape [batch, channels, height, width].
    kernel : Tuple[int, int]
        Tuple specifying the height and width of the pooling kernel.

    Returns
    -------
    Tensor
        Tensor after applying max pooling, with reduced height and width
        depending on the kernel size.
        Shape: [batch, channels, new_height, new_width]

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, dim=4)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout regularization to the input tensor.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    p : float, optional
        Dropout probability (default is 0.5).
    ignore : bool, optional
        If True, bypass dropout (default is False).

    Returns
    -------
    Tensor
        Tensor with randomly zeroed elements scaled by 1 / (1 - p) to maintain expected value.

    """
    if p == 1.0:
        if not ignore:
            return input.zeros(input.shape)
        else:
            return input
    if ignore:
        return input
    mask = rand(input.shape, backend=input.backend) > p
    return input * mask * (1.0 / (1 - p))
