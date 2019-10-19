"""Polynomial base class."""
from __future__ import division
import re
from six import string_types

import numpy

from . import align, construct, array_function, poly_function


REDUCE_MAPPINGS = {
    numpy.add: numpy.sum,
    numpy.multiply: numpy.prod,
    numpy.logical_and: numpy.all,
    numpy.logical_or: numpy.any,
}
ACCUMULATE_MAPPINGS = {
    numpy.add: numpy.cumsum,
    numpy.multiply: numpy.cumprod,
}
INDETERMINANT_DEFAULT = "q"
"""Polynomial indeterminant defaults, if not defined."""

INDETERMINANT_DEFAULT_INDEX = False
"""
Add a postfix index to single indeterminant name.

If single indeterminant name, e.g. 'q' is provided, but the polynomial is
multivariate, an extra postfix index is added to differentiate the names:
'q0, q1, q2, ...'. If true, encorce this behavior for single variables as
well such that 'q' always get converted to 'q0'.
"""

INDETERMINANT_REGEX = r"[\w_]+"
"""Regular expression definint valid indeterminant names."""


class ndpoly(numpy.ndarray):  # pylint: disable=invalid-name
    """
    Polynomial as numpy array.

    An array object represents a multidimensional, homogeneous polynomial array
    of fixed-size items. An associated data-type object describes the format of
    each element in the array (its byte-order, how many bytes it occupies in
    memory, whether it is an integer, a floating point number, or something
    else, etc.)

    Arrays should be constructed using `symbols`, `monomials`, `polynomial`,
    etc.

    Attributes:
        coefficients (List[numpy.ndarray, ...]):
            The polynomial coefficients. Together with exponents defines the
            polynomial form.
        exponents (numpy.ndarray):
            The polynomial exponents. 2-dimensionsl where the first axis is the
            same length as coefficients and the second is the length of the
            indeterminants.
        keys (List[str, ...]):
            The raw names of the coefficients. One-to-one with `exponents`, but
            as string as to be compatible with numpy structured array. Unlike
            the exponents, that are useful for mathematical manipulation, the
            keys are useful as coefficient lookup.
        indeterminats (numpoly.ndpoly):
            Secondary polynomial only consisting of an array of simple
            independent variables found in the polynomial array.
        names (Tuple[str, ...]):
            Same as `indeterminants`, but only the names as string.
        values (numpy.ndarray):
            Expose the underlying structured array.

    Examples:
        >>> poly = ndpoly(
        ...     exponents=[(0, 1), (0, 0)], shape=(3,), indeterminants="x y")
        >>> poly["00"] = 1, 2, 3
        >>> poly["01"] = 4, 5, 6
        >>> print(numpy.array(list(poly.coefficients)))
        [[4 5 6]
         [1 2 3]]
        >>> print(poly)
        [4*y+1 5*y+2 6*y+3]
        >>> print(poly[0])
        4*y+1

    """

    # =================================================
    # Stuff to get subclassing of ndarray to run smooth
    # =================================================

    __array_priority__ = 16
    _dtype = None
    keys = None
    names = None

    def __new__(
            cls,
            exponents=((0,),),
            shape=(),
            indeterminants=None,
            dtype=None,
            **kwargs
    ):
        """
        Class constructor.

        Args:
            exponents (numpy.ndarray):
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            shape (Tuple[int, ...]):
                Shape of created array.
            indeterminants (Union[None, str, Tuple[str], numpoly.ndpoly]):
                The name of the indeterminant variables in te polynomial. If
                polynomial, inherent from it. Else, pass argument to
                `numpoly.symbols` to create the indeterminants. If only one
                name is provided, but more than one is required, indeterminant
                will be extended with an integer index. If omitted, use
                ``INDETERMINANT_DEFAULT``.
            dtype:
                Any object that can be interpreted as a numpy data type.
            kwargs:
                Extra arguments passed to `numpy.ndarray` constructor.

        """
        exponents = numpy.array(exponents, dtype=numpy.uint32)

        if indeterminants is None:
            indeterminants = INDETERMINANT_DEFAULT
        if isinstance(indeterminants, str):
            indeterminants = poly_function.symbols(indeterminants)
        if isinstance(indeterminants, ndpoly):
            indeterminants = indeterminants.names
        if len(indeterminants) == 1 and (INDETERMINANT_DEFAULT_INDEX or
                                         exponents.shape[1] > 1):
            indeterminants = tuple(
                "%s%d" % (str(indeterminants[0]), idx)
                for idx in range(exponents.shape[1])
            )
        for indeterminant in indeterminants:
            assert re.search(INDETERMINANT_REGEX, indeterminant), (
                "invalid polynomial name; expected format: '%s'" % INDETERMINANT_REGEX)

        keys = (exponents+48).flatten().view("U%d" % exponents.shape[-1])

        dtype = int if dtype is None else dtype
        dtype_ = numpy.dtype([(key, dtype) for key in keys])

        obj = super(ndpoly, cls).__new__(
            cls, shape=shape, dtype=dtype_, **kwargs)
        obj._dtype = numpy.dtype(dtype)  # pylint: disable=protected-access
        obj.keys = keys  # pylint: disable=protected-access
        obj.names = tuple(indeterminants)  # pylint: disable=protected-access
        return obj

    def __array_finalize__(self, obj):
        """Finalize numpy constructor."""
        if obj is None:
            return
        self.keys = getattr(obj, "keys", None)
        self.names = getattr(obj, "names", None)
        self._dtype = getattr(obj, "_dtype", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatch method for operators."""
        if method == "reduce":
            ufunc = REDUCE_MAPPINGS[ufunc]
        elif method == "accumulate":
            ufunc = ACCUMULATE_MAPPINGS[ufunc]
        else:
            assert method == "__call__", (
                "method %s not recognised" % method)
        assert ufunc in array_function.ARRAY_FUNCTIONS, (
            "function %s not supported by numpoly." % ufunc)
        return array_function.ARRAY_FUNCTIONS[ufunc](*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatch method for functions."""
        assert func in array_function.ARRAY_FUNCTIONS, (
            "function %s not supported by numpoly." % func)
        return array_function.ARRAY_FUNCTIONS[func](*args, **kwargs)

    # ======================================
    # Properties specific for ndpoly objects
    # ======================================

    @property
    def coefficients(self):
        """
        Polynomial coefficients.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(numpy.array(poly.coefficients))
            [[ 0 14]
             [ 0 -3]
             [ 2  0]]

        """
        out = numpy.empty((len(self.keys),) + self.shape, dtype=self._dtype)
        for idx, key in enumerate(self.keys):
            out[idx] = numpy.ndarray.__getitem__(self, key)
        return list(out)

    @property
    def exponents(self):
        """
        Polynomial exponents.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(poly.exponents)
            [[0 0]
             [0 2]
             [4 0]]

        """
        exponents = self.keys.flatten().view(numpy.uint32)-48
        return exponents.reshape(len(self.keys), -1)

    @staticmethod
    def from_attributes(
            exponents,
            coefficients,
            indeterminants="q",
            dtype=None,
            clean=True,
    ):
        """
        Construct polynomial from polynomial attributes.

        Args:
            exponents (numpy.ndarray):
                The exponents in an integer array with shape ``(N, D)``, where
                ``N`` is the number of terms in the polynomial sum and ``D`` is
                the number of dimensions.
            coefficients (Iterable[numpy.ndarray]):
                The polynomial coefficients. Must correspond to `exponents` by
                having the same length ``N``.
            indeterminants (Union[str, Tuple[str, ...], numpoly.ndpoly]):
                The indeterminants variables, either as string names or as
                simple polynomials. Must correspond to the exponents by having
                length ``D``.
            dtype (Optional[numpy.dtype]):
                The data type of the polynomial. If omitted, extract from
                `coefficients`.
            clean (bool):
                Clean up attributes, removing redundant indeterminants and
                exponents. Used to ensure alignment isn't broken.

        Returns:
            (numpoly.ndpoly):
                Polynomials with attributes defined by input.

        Examples:
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0]], coefficients=[4])
            polynomial(4)
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[1]], coefficients=[[1, 2, 3]])
            polynomial([q, 2*q, 3*q])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0], [1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q, 1+q])
            >>> numpoly.ndpoly.from_attributes(
            ...     exponents=[[0, 1], [1, 1]], coefficients=[[0, 1], [1, 1]])
            polynomial([q0*q1, q1+q0*q1])

        """
        return construct.polynomial_from_attributes(
            exponents=exponents,
            coefficients=coefficients,
            indeterminants=indeterminants,
            dtype=dtype,
            clean=clean,
        )

    @property
    def indeterminants(self):
        """
        Polynomial indeterminants.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([2*x**4, -3*y**2+14])
            >>> print(poly)
            [2*x**4 14-3*y**2]
            >>> print(poly.indeterminants)
            [x y]

        """
        return construct.polynomial_from_attributes(
            exponents=numpy.eye(len(self.names), dtype=int),
            coefficients=numpy.eye(len(self.names), dtype=int),
            indeterminants=self.names,
        )

    @property
    def values(self):
        """
        Expose the underlying structured array.

        Typically used for operator dispatching and not for use to conversion.
        For that, see :func:`numpoly.tonumpy`.

        Examples:
            >>> numpoly.symbols("x").values
            array((1,), dtype=[('1', '<i8')])
            >>> numpoly.symbols("x y").values
            array([(1, 0), (0, 1)], dtype=[('10', '<i8'), ('01', '<i8')])

        """
        return numpy.ndarray(
            shape=self.shape,
            dtype=[(key, self.dtype) for key in self.keys],
            buffer=self.data
        )

    def isconstant(self):
        """
        Check if a polynomial is constant or not.

        Returns:
            (bool):
                True if all elements in array are constant.

        Examples:
            >>> x = numpoly.symbols("x")
            >>> x.isconstant()
            False
            >>> numpoly.polynomial([1, 2]).isconstant()
            True
            >>> numpoly.polynomial([1, x]).isconstant()
            False

        """
        return poly_function.isconstant(self)

    def todict(self):
        """
        Cast to dict where keys are exponents and values are coefficients.

        Returns:
            (Dict[Tuple[int, ...], numpy.ndarray]):
                Dictionary where keys are exponents and values are
                coefficients.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = 2*x**4-3*y**2+14
            >>> print(poly)
            14+2*x**4-3*y**2
            >>> poly.todict() == {(0, 0): 14, (4, 0): 2, (0, 2): -3}
            True

        """
        return {tuple(exponent): coefficient
                for exponent, coefficient in zip(
                    self.exponents, self.coefficients)}

    def tonumpy(self):
        """
        Cast polynomial to numpy.ndarray, if possible.

        Raises:
            ValueError:
                When polynomial include indeterminats, casting to numpy.

        Returns:
            (numpy.ndarray):
                Same as object, but cast to `numpy.ndarray`.

        Examples:
            >>> numpoly.polynomial([1, 2]).tonumpy()
            array([1, 2])
            >>> numpoly.symbols("x").tonumpy()
            Traceback (most recent call last):
                ...
            ValueError: only constant polynomials can be converted to array.

        """
        return poly_function.tonumpy(self)

    # =============================================
    # Override numpy properties to work with ndpoly
    # =============================================

    @property
    def dtype(self):
        """Show coefficient dtype instead of the structured array."""
        return self._dtype

    def astype(self, dtype, **kwargs):
        """Wrap ndarray.astype."""
        coefficients = [coefficient.astype(dtype, **kwargs)
                        for coefficient in self.coefficients]
        return construct.polynomial_from_attributes(
            self.exponents, coefficients, self.names)

    def round(self, decimals=0, out=None):
        """Wrap ndarray.round."""
        # Not sure why it is required. Likely a numpy bug.
        return array_function.around(self, decimals=decimals, out=out)

    # ============================================================
    # Override dunder methods that isn't dealt with by dispatching
    # ============================================================

    def __call__(self, *args, **kwargs):
        """Evaluate polynomial."""
        return poly_function.call(self, *args, **kwargs)

    def __eq__(self, other):
        """Left equality."""
        return array_function.equal(self, other)

    def __getitem__(self, index):
        """
        Get array item or slice.

        Args:
            index (Union[int, str, Tuple[int, ...], numpy.ndarray]):
                The index to extract. If string type, extract using th
                underlying structured array data type. Else, extract as
                numpy.ndarray.

        Examples:
            >>> x, y = numpoly.symbols("x y")
            >>> poly = numpoly.polynomial([[1-4*x, x**2], [y-3, x*y*y]])
            >>> print(poly)
            [[1-4*x x**2]
             [-3+y x*y**2]]
            >>> print(poly[0])
            [1-4*x x**2]
            >>> print(poly[:, 1])
            [x**2 x*y**2]
            >>> print(poly["10"])
            [[-4  0]
             [ 0  0]]

        """
        if isinstance(index, string_types):
            return numpy.asarray(super(ndpoly, self).__getitem__(index))
        return construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=[coeff[index] for coeff in self.coefficients],
            indeterminants=self.names,
        )

    def __iter__(self):
        """Iterate polynomial array."""
        coefficients = numpy.array(list(self.coefficients))
        return iter(construct.polynomial_from_attributes(
            exponents=self.exponents,
            coefficients=coefficients[:, idx],
            indeterminants=self.names,
        ) for idx in range(len(self)))

    def __ne__(self, other):
        """Not equal."""
        return array_function.not_equal(self, other)

    def __repr__(self):
        """Canonical string representation."""
        return array_function.array_repr(self)

    def __str__(self):
        """Pretty string representation."""
        return array_function.array_str(self)
