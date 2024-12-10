"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negative of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the negation function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the inverse operation.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise addition of two tensors.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor
            The element-wise sum of the input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the addition operation.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Gradients of the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise multiplication of two tensors.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor
            The element-wise product of the input tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the multiplication operation.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Gradients of the input tensors.

        """
        (t1, t2) = ctx.saved_values
        return grad_output.f.mul_zip(t2, grad_output), grad_output.f.mul_zip(
            t1, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the sigmoid of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The sigmoid-transformed tensor.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the sigmoid function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the ReLU (rectified linear unit) of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The ReLU-transformed tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the ReLU function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the natural logarithm of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The log-transformed tensor.

        """
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the log function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the element-wise exponential of a tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The exponential of the input tensor.

        """
        exp_tensor = t1.f.exp_map(t1)
        ctx.save_for_backward(exp_tensor)
        return exp_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the exponential function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tensor
            Gradient of the input tensor.

        """
        (exp_tensor,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp_tensor)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum of elements of a tensor along a specific dimension.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.
        dim : Tensor
            The dimension along which the sum is computed.

        Returns
        -------
        Tensor
            The sum of elements along the specified dimension.

        """
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the sum function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, float]
            Gradient of the input tensor and a constant.

        """
        (t1_shape, dim) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute element-wise less-than comparison between two tensors.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor
            A tensor containing the element-wise result of t1 < t2.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return zero gradients as LT is non-differentiable.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Zero gradients.

        """
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute element-wise equality comparison between two tensors.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor
            A tensor containing the element-wise result of t1 == t2.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return zero gradients as EQ is non-differentiable.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Zero gradients.

        """
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute element-wise "is close" comparison between two tensors.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor
            A tensor containing the element-wise result of t1 being close to t2.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of the input tensor.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        t1 : Tensor
            The input tensor.
        order : Tensor
            The order of permutation.

        Returns
        -------
        Tensor
            The permuted tensor.

        """
        ctx.save_for_backward(order)
        return t1._new(t1._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the permute function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, float]
            Permuted gradient of the input tensor and a constant.

        """
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor into the specified shape.

        Parameters
        ----------
        ctx : Context
            Context object for storing information during the forward pass.
        a : Tensor
            The input tensor.
        shape : Tensor
            The new shape.

        Returns
        -------
        Tensor
            The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the view (reshape) function.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        grad_output : Tensor
            Upstream gradient.

        Returns
        -------
        Tuple[Tensor, float]
            Reshaped gradient of the input tensor and a constant.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient of the function `f` with respect to its argument using
    the central difference approximation.
    This method estimates the partial derivative of `f` with respect to `arg` at a
    given index `ind` by computing the finite difference of `f` at two nearby points,
    shifted by `epsilon`.

    Parameters
    ----------
    f : Callable
        The function whose gradient is to be computed.
    vals : Tensor
        The input tensors to the function `f`.
    arg : int, optional
        The index of the argument with respect to which the gradient will be computed
        (default is 0).
    epsilon : float, optional
        A small perturbation used to compute the finite difference (default is 1e-6).
    ind : UserIndex
        The index at which the gradient will be evaluated.

    Returns
    -------
    float
        The computed gradient with respect to the argument at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
