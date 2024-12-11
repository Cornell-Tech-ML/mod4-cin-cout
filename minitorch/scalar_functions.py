from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the forward pass of a function to the input Scalar values and return a new Scalar with a history.

        Parameters
        ----------
        cls : type
            The class of the function being applied.
        *vals : ScalarLike
            Input values, which can be either Scalars or floats, to apply the function to.

        Returns
        -------
        Scalar
            A new Scalar instance resulting from applying the function, along with its history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two values.

        Parameters
        ----------
        ctx : Context
            Context object to save information for backpropagation.
        a : float
            The first value.
        b : float
            The second value.

        Returns
        -------
        float
            The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the addition function with respect to a and b.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        d_output : float
            Upstream gradient.

        Returns
        -------
        Tuple[float, ...]
            The gradients with respect to a and b.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a value.

        Parameters
        ----------
        ctx : Context
            Context object to save information for backpropagation.
        a : float
            Input value.

        Returns
        -------
        float
            The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the log function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of two values.

        Parameters
        ----------
        ctx : Context
            Context object to save information for backpropagation.
        a : float
            The first value.
        b : float
            The second value.

        Returns
        -------
        float
            The product of a and b.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the multiplication function with respect to a and b.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        Tuple[float, float]
            The gradients with respect to a and b.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of a value.

        Parameters
        ----------
        ctx : Context
            Context object to save information for backpropagation.
        a : float
            The input value.

        Returns
        -------
        float
            The inverse of a (1/a).

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negate the input value.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            Input value.

        Returns
        -------
        float
            The negated value (-a).

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of the input value.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            Input value.

        Returns
        -------
        float
            Sigmoid value of a.

        """
        sigmoid_value = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_value)
        return sigmoid_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the sigmoid function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        sigmoid_value: float = ctx.saved_values[0]
        return sigmoid_value * (1 - sigmoid_value) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the ReLU function to the input value.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            Input value.

        Returns
        -------
        float
            ReLU of a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the ReLU function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of the input value.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            Input value.

        Returns
        -------
        float
            Exponential value of a.

        """
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the exponential function with respect to a.

        Parameters
        ----------
        ctx : Context
            Context object containing saved values from the forward pass.
        d_output : float
            Upstream gradient.

        Returns
        -------
        float
            The gradient with respect to a.

        """
        exp_value: float = ctx.saved_values[0]
        return d_output * exp_value


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute whether a is less than b.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            The first value.
        b : float
            The second value.

        Returns
        -------
        float
            1.0 if a < b, 0.0 otherwise.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Return zero gradients as LT is non-differentiable.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        d_output : float
            Upstream gradient.

        Returns
        -------
        Tuple[float, float]
            Zero gradients.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute whether a is equal to b.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        a : float
            The first value.
        b : float
            The second value.

        Returns
        -------
        float
            1.0 if a == b, 0.0 otherwise.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Return zero gradients as EQ is non-differentiable.

        Parameters
        ----------
        ctx : Context
            Context object for backpropagation.
        d_output : float
            Upstream gradient.

        Returns
        -------
        Tuple[float, float]
            Zero gradients.

        """
        return 0.0, 0.0
