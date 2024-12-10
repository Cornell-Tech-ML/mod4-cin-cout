"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# Multiplies two numbers
def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        The product of x and y.

    """
    return x * y


# Returns the input unchanged
def id(x: float) -> float:
    """Return the input unchanged.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The same input number.

    """
    return x


# Adds two numbers
def add(x: float, y: float) -> float:
    """Add two numbers.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        The sum of x and y.

    """
    return x + y


# Negates a number
def neg(x: float) -> float:
    """Negate a number.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The negation of x.

    """
    return -x


# Checks if one number is less than another
def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        1.0 if x is less than y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


# Checks if two numbers are equal
def eq(x: float, y: float) -> float:
    """Check if x is equal to y.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        1.0 if x is equal to y, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


# Returns the larger of two numbers
def max(x: float, y: float) -> float:
    """Return the larger of two numbers.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        The larger number.

    """
    return x if x > y else y


# Checks if two numbers are close in value
def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close in value.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    bool
        True if the absolute difference between x and y is less than 1e-2.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


# Calculates the sigmoid function
def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# Applies the ReLU activation function
def relu(x: float) -> float:
    """Apply the ReLU activation function.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The ReLU of x (max(0, x)).

    """
    return x if x > 0 else 0.0


ESP = 1e-6


# Calculates the natural logarithm
def log(x: float) -> float:
    """Compute the natural logarithm of a number.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The natural logarithm of x.

    """
    return math.log(x + ESP)


# Calculates the exponential function
def exp(x: float) -> float:
    """Compute the exponential of a number.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The exponential of x.

    """
    return math.exp(x)


# Calculates the reciprocal
def inv(x: float) -> float:
    """Compute the reciprocal of a number.

    Parameters
    ----------
    x : float
        Input number.

    Returns
    -------
    float
        The reciprocal of x (1/x).

    """
    return 1.0 / x


# Computes the derivative of log times a second argument
def log_back(x: float, d: float) -> float:
    """Compute the gradient of the log function with respect to x.

    Parameters
    ----------
    x : float
        Input number.
    d : float
        Upstream gradient.

    Returns
    -------
    float
        Gradient of log(x) with respect to x.

    """
    return d / x


# Computes the derivative of reciprocal times a second argument
def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the reciprocal function with respect to x.

    Parameters
    ----------
    x : float
        Input number.
    d : float
        Upstream gradient.

    Returns
    -------
    float
        Gradient of 1/x with respect to x.

    """
    return -d / (x**2)


# Computes the derivative of ReLU times a second argument
def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function with respect to x.

    Parameters
    ----------
    x : float
        Input number.
    d : float
        Upstream gradient.

    Returns
    -------
    float
        Gradient of ReLU(x) with respect to x.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


# Applies a given function to each element in an iterable
# Higher-order function that applies a given function to each element of an iterable
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in an iterable.

    Parameters
    ----------
    fn : Callable[[float], float]
        Function to apply.
    lst : Iterable[float]
        List of input values.

    Returns
    -------
    Iterable[float]
        List of output values.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


# Combines elements from two iterables using a given function
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to pairs of elements from two iterables.

    Parameters
    ----------
    fn : Callable[[float, float], float]
        Function to apply.
    lst1 : Iterable[float]
        First list of input values.
    lst2 : Iterable[float]
        Second list of input values.

    Returns
    -------
    Iterable[float]
        List of output values.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# Reduces an iterable to a single value using a given function
def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value using a function.

    Parameters
    ----------
    fn : Callable[[float, float], float]
        Function to apply.
    lst : Iterable[float]
        List of input values.
    start : float
        Initial value for the reduction.

    Returns
    -------
    float
        The reduced value.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# Negates all elements in a list using map
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Parameters
    ----------
    ls : Iterable[float]
        List of input values.

    Returns
    -------
    Iterable[float]
        List of negated values.

    """
    return map(neg)(ls)


# Adds corresponding elements from two lists using zipWith
def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Parameters
    ----------
    lst1 : Iterable[float]
        First list of input values.
    lst2 : Iterable[float]
        Second list of input values.

    Returns
    -------
    Iterable[float]
        List of summed values.

    """
    return zipWith(add)(lst1, lst2)


# Sums all elements in a list using reduce
def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list.

    Parameters
    ----------
    lst : Iterable[float]
        List of input values.

    Returns
    -------
    float
        The sum of the elements.

    """
    return reduce(add, 0.0)(lst)


# Calculates the product of all elements in a list using reduce
def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Parameters
    ----------
    lst : Iterable[float]
        List of input values.

    Returns
    -------
    float
        The product of the elements.

    """
    return reduce(mul, 1.0)(lst)
