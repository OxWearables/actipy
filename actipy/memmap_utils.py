import warnings
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


def _check_len(xs):
    n0 = len(xs[0])
    for x in xs:
        n = len(x)
        assert n0 == n, f"Arrays must be equal lenghts, but are {n0} and {n}"


def _check_shape(xs, ignore_first_axis=True):
    shape0 = xs[0].shape
    for x in xs:
        shape = x.shape
        if not ignore_first_axis:
            assert shape0 == shape, f"Arrays must be equal shape, but are {shape0} and {shape}"
        else:
            if len(shape0) > 1:
                assert len(shape) > 1 and shape0[1:] == shape[1:], f"Arrays must be equal shape except for first axis, but are {shape0} and {shape}"


# def apply(fn, *xs, chunksize=100_000, leeway=0):
#     """ Apply f(*xs) chunk by chunk. A leeway parameter can be passed to
#     allow for overlapping chunks. The input arrays for fn must have equal
#     lengths. The result is returned as a memmap array.
#     """

#     y = concat([copy(chunk) for chunk in chunker(*xs, chunksize=chunksize, leeway=leeway, fn=fn)])

#     return y


# def chunker(*xs, chunksize=100_000, leeway=0, fn=None, fntrim=True):
#     """ Return chunk generator for given numpy array(s).
#     A `leeway` parameter can be used to obtain overlapping chunks (e.g. leeway='30m').
#     If a function `fn` is provided, it is applied to each chunk(s). The leeway is
#     trimmed after function application by default (set `fntrim=False` to skip).
#     """

#     _check_len(xs)

#     n = len(xs[0])

#     for i in range(0, n, chunksize):
#         csize = min(chunksize, n - i)
#         leeway_left = min(i, leeway)
#         leeway_right = min(leeway, n - (i + csize))
#         chunk = tuple(x[i - leeway_left : i + csize + leeway_right] for x in xs)

#         if fn is not None:
#             chunk = fn(*chunk)

#             if leeway > 0 and fntrim:
#                 try:
#                     if leeway_right > 0:
#                         chunk = chunk[leeway_left : - leeway_right]
#                     else:
#                         chunk = chunk[leeway_left:]
#                 except Exception:
#                     warnings.warn(f"Could not trim chunk. Ignoring fntrim={fntrim}...")

#             yield chunk
#             continue

#         if len(chunk) == 1:  # undo the tuple if single element
#             chunk = chunk[0]

#         yield chunk
