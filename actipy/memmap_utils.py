import tempfile
import numpy as np



def concat(xs):
    """ Concatenate a list of arrays. Only concatenation along the first axis is
    currently supported. The concatenated result is returned as a memmap array.
    """

    _check_dtype(xs)
    _check_shape(xs)

    # determine final shape
    n = sum(len(x) for x in xs)
    shape = xs[0].shape
    if len(shape) > 1:
        shape = (n,) + shape[1:]
    else:
        shape = (n,)

    dtype = xs[0].dtype

    with tempfile.NamedTemporaryFile() as f:
        y = np.memmap(f.name, mode="w+", dtype=dtype, shape=shape)

    # concatenate
    i = 0
    for x in xs:
        m = len(x)
        y[i:i + m] = x
        i += m

    return y


def empty(shape, dtype):
    """ Return an empty memmap array with specified shape and dtype. """
    with tempfile.NamedTemporaryFile() as f:
        y = np.memmap(f.name, mode="w+", dtype=dtype, shape=shape)
    return y


def empty_like(x):
    """ Return an empty memmap array with same shape and dtype as the given array. """
    dtype = x.dtype
    shape = x.shape
    y = empty(shape, dtype)
    return y


def copy(x):
    """ Return a memmap array copy of the given array. """
    y = empty_like(x)
    y[:] = x
    return y


def _check_dtype(xs):
    dtype0 = xs[0].dtype
    for x in xs:
        dtype = x.dtype
        assert dtype0 == dtype, f"Arrays must be of equal dtype, but are {dtype0} and {dtype}"


def _check_shape(xs, ignore_first_axis=True):
    shape0 = xs[0].shape
    for x in xs:
        shape = x.shape
        if not ignore_first_axis:
            assert shape0 == shape, f"Arrays must be equal shape, but are {shape0} and {shape}"
        else:
            if len(shape0) > 1:
                assert len(shape) > 1 and shape0[1:] == shape[1:], f"Arrays must be equal shape except for first axis, but are {shape0} and {shape}"
