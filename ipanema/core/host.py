################################################################################
#                                                                              #
#                        OPERATIONS WITH PYTHON BACKEND                        #
#                                                                              #
################################################################################

from . import types
import numpy as np

__all__ = ['arange']

################################################################################
# Ristra functions #############################################################

def arange(n, dtype=types.cpu_int):
  if dtype == types.cpu_int:
    return np.arange(n, dtype=dtype)
  elif dtype == types.cpu_complex:
    return np.arange(n, dtype=dtype).astype(types.cpu_complex)
  else:
    raise NotImplementedError(f'Not implemented for data type "{dtype}"')



def ale(a1, a2):
    return a1 < a2


# create a n-dimensional mesh
def ndmesh(*args):
   args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])



def concatenate(arrays, maximum=None):
  if maximum is not None:
    return np.concatenate(arrays)[:maximum]
  else:
    return np.concatenate(arrays)



def count_nonzero(a):
  return np.count_nonzero(a)



def allocate(a, copy=True, convert=True):
  if convert:
    if a.dtype != types.cpu_type:
      a = a.astype(types.cpu_type)
    return a
  if copy:
    return a.copy()
  else:
    return a



def empty(size, dtype=types.cpu_type):
  return np.empty(size, dtype=dtype)



def exp(a):
  return np.exp(a)



def get(a):
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



def log10(a):
  return np.log10(a)



def log2(a):
  return np.log2(a)



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
    return np.zeros(n, dtype=types.cpu_real_bool)
  else:
    return np.zeros(n, dtype=dtype)
