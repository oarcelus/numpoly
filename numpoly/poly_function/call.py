"""Evaluate polynomial by inserting new values in to the indeterminants."""

from __future__ import annotations
from typing import Dict, Sequence, Optional, Union, List
import logging

import numpy
import numpoly

from ..baseclass import ndpoly, PolyLike
import cProfile
import io
import pstats
import time


def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats()
        print(stream.getvalue())
        return result

    return wrapper


# @profile
def call(
    poly: PolyLike,
    args: Sequence[Optional[PolyLike]] = (),
    kwargs: Dict[str, PolyLike] = None,
) -> Union[numpy.ndarray, ndpoly]:
    """
    Evaluate polynomial by inserting new values in to the indeterminants.

    Equivalent to calling the polynomial or using the ``__call__`` method.

    Args:
        poly:
            Polynomial to evaluate.
        args:
            Argument to evaluate indeterminants. Ordered positional by
            ``poly.indeterminants``. None values indicate that a variable is
            not to be evaluated, creating a partial evaluation.
        kwargs:
            Same as ``args``, but positioned by name.

    Return:
        Evaluated polynomial. If the resulting array does not contain any
        indeterminants, an array is returned instead of a polynomial.

    Example:
        >>> q0, q1 = numpoly.variable(2)
        >>> poly = numpoly.polynomial([[q0, q0-1], [q1, q1+q0]])
        >>> numpoly.call(poly)
        polynomial([[q0, q0-1],
                    [q1, q1+q0]])
        >>> poly
        polynomial([[q0, q0-1],
                    [q1, q1+q0]])
        >>> numpoly.call(poly, (1, 0))
        array([[1, 0],
               [0, 1]])
        >>> numpoly.call(poly, (1,), {"q1": [0, 1, 2]})
        array([[[1, 1, 1],
                [0, 0, 0]],
        <BLANKLINE>
               [[0, 1, 2],
                [1, 2, 3]]])
        >>> numpoly.call(poly, (q1,))
        polynomial([[q1, q1-1],
                    [q1, 2*q1]])
        >>> numpoly.call(poly, kwargs={"q1": q0-1, "q0": 2*q1})
        polynomial([[2*q1, 2*q1-1],
                    [q0-1, 2*q1+q0-1]])

    """
    logger = logging.getLogger(__name__)

    poly = numpoly.aspolynomial(poly)
    kwargs = kwargs if kwargs else {}

    parameters = dict(zip(poly.names, poly.indeterminants))
    if kwargs:
        parameters.update(kwargs)
    for arg, name in zip(args, poly.names):
        if name in kwargs:
            raise TypeError(f"multiple values for argument '{name}'")
        if arg is not None:
            parameters[name] = arg
    extra_args = [key for key in parameters if key not in poly.names]
    if extra_args:
        raise TypeError(f"unexpected keyword argument '{extra_args[0]}'")

    # There can only be one shape:
    ones = numpy.ones((), dtype=int)
    for value in parameters.values():
        ones = ones * numpy.ones(numpoly.polynomial(value).shape, dtype=int)
    shape = poly.shape + ones.shape

    logger.debug("poly shape: %s", poly.shape)
    logger.debug("parameter common shape: %s", ones.shape)
    logger.debug("output shape: %s", shape)
    # main loop:
    allnumpy = all(type(parameters[name]) is numpy.ndarray for name in poly.names)
    if allnumpy:
        allshape = all(
            parameters[name].shape == parameters[poly.names[0]].shape
            for name in poly.names
        )
        if allshape and poly.ndim >= 1:
            out = numpoly.ccall(poly, ones, parameters, shape)
            if isinstance(out, numpoly.ndpoly):
                if out.isconstant():
                    out = out.tonumpy()
                else:
                    out, _ = numpoly.align_indeterminants(out, poly.indeterminants)

            return out

    out = numpy.zeros((), dtype=int)
    for exponent, coefficient in zip(poly.exponents, poly.coefficients):
        term = ones
        for power, name in zip(exponent, poly.names):
            term = term * parameters[name] ** power

        if isinstance(term, numpoly.ndpoly):
            tmp = numpoly.outer(coefficient, term)
        else:
            tmp = numpy.outer(coefficient, term)
        out = out + tmp.reshape(shape)

    if isinstance(out, numpoly.ndpoly):
        if out.isconstant():
            out = out.tonumpy()
        else:
            out, _ = numpoly.align_indeterminants(out, poly.indeterminants)

    return out
