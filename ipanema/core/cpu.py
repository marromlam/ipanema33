'''
Operations with numpy objects
'''
from . import docstrings
from . import types

import numpy as np


def arange(n, dtype=types.cpu_int):
    if dtype == types.cpu_int:
        return np.arange(n, dtype=dtype)
    elif dtype == types.cpu_complex:
        return np.arange(n, dtype=dtype).astype(types.cpu_complex)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')


def ale(a1, a2):
    return a1 < a2


def concatenate(arrays, maximum=None):
    if maximum is not None:
        return np.concatenate(arrays)[:maximum]
    else:
        return np.concatenate(arrays)


def count_nonzero(a):
    return np.count_nonzero(a)


def data_array(a, copy=True, convert=True):
    if copy:
        return np.array(a, dtype=types.cpu_type)
    else:
        if a.dtype != types.cpu_type:
            return a.astype(types.cpu_type)
        return a


def empty(size, dtype=types.cpu_type):
    return np.empty(size, dtype=dtype)


def exp(a):
    return np.exp(a)


def extract_ndarray(a):
    return a


def false_till(N, n):
    a = np.zeros(N, dtype=types.cpu_real_bool)
    a[n:] = True
    return a


def fft(a):
    return np.fft.fft(a)


def fftconvolve(a, b, data):

    fa = fft(a)
    fb = fft(b)

    shift = fftshift(data)

    output = ifft(fa * shift * fb)

    return output * (data[1] - data[0])


def fftshift(a):
    n0 = sum(a < 0)
    nt = len(a)
    com = types.cpu_complex(+2.j * np.pi * n0 / nt)
    rng = arange(nt, dtype=types.cpu_int).astype(types.cpu_complex)
    return exp(com * rng)


def geq(a, v):
    return a >= v


def ifft(a):
    return np.fft.ifft(a)


def interpolate_linear(x, xp, yp):
    return np.interp(x, xp, yp)


def le(a, v):
    return a < v


def leq(a, v):
    return a <= v


def linspace(vmin, vmax, size):
    return np.linspace(vmin, vmax, size, dtype=types.cpu_type)


def log(a):
    return np.log(a)


def logical_and(a, b):
    return np.logical_and(a, b)


def logical_or(a, b):
    return np.logical_or(a, b)


def max(a):
    return np.max(a)


def meshgrid(*arrays):
    return tuple(map(np.ndarray.flatten, np.meshgrid(*arrays)))


def min(a):
    return np.min(a)


def ones(n, dtype=types.cpu_type):
    if dtype == types.cpu_bool:
        # Hack due to lack of "bool" in PyOpenCL
        return np.ones(n, dtype=types.cpu_real_bool)
    else:
        return np.ones(n, dtype=dtype)


def random_uniform(vmin, vmax, size):
    return np.random.uniform(vmin, vmax, size)


def real(a):
    return a.real


def shuffling_index(n):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


def sum(a, *args):
    if len(args) == 0:
        return np.sum(a)
    else:
        return np.sum((a, *args), axis=0)


def sum_inside(centers, edges, values=None):
    out, _ = np.histogramdd(centers, bins=edges, weights=values)
    return out.flatten()


def slice_from_boolean(a, valid):
    return a[valid]


def slice_from_integer(a, indices):
    return a[indices]


def true_till(N, n):
    a = np.ones(N, dtype=types.cpu_real_bool)
    a[n:] = False
    return a


def zeros(n, dtype=types.cpu_type):
    if dtype == types.cpu_bool:
        # Hack due to lack of "bool" in PyOpenCL
        return np.zeros(n, dtype=types.cpu_real_bool)
    else:
        return np.zeros(n, dtype=dtype)
