################################################################################
#                                                                              #
#                        OPERATIONS WITH DEVICE BACKEND                        #
#                                                                              #
################################################################################

import builtins
import reikna
from reikna.cluda import functions
from reikna.fft import FFT
import functools
import numpy as np
import os
import sys
import threading

from . import multi_par
from . import types
from .utils import get_sizes

THREAD = builtins.THREAD

################################################################################
# CACHE ########################################################################

# Save the FFT compiled objects
FFT_CACHE = {}


class ArrayCacheManager(object):

    def __init__(self, dtype):
        '''
        Object that keeps array in the GPU device in order to avoid creating
        and destroying them many times, and calls functions with them.

        :param dtype: data type of the output arrays.
        :type dtype: numpy.dtype
        '''
        self.__cache = {}
        self.__dtype = dtype
        self.__lock = threading.Lock()
        super(ArrayCacheManager, self).__init__()

    def free_cache(self):
        '''
        Free the cache of this object, removing the arrays not being used.
        '''
        with self.__lock:

            # Process the arrays to look for those not being used
            remove = []
            for s, elements in self.__cache.items():
                for i, el in reversed(enumerate(elements)):
                    if sys.getrefcount(el) == 1:
                        remove.append((s, i))
            for s, i in remove:
                self.__cache[s].pop(i)

            # Clean empty lists
            remove = []
            for s, lst in self.__cache.items():
                if len(lst) == 0:
                    remove.append(s)

            for s in remove:
                self.__cache.pop(s)

    def get_array(self, size):
        '''
        Get the array with size "size" from the cache, if it exists.

        :param size: size of the output array.
        :type size: int
        '''
        with self.__lock:

            elements = self.__cache.get(size, None)

            if elements is not None:
                for el in elements:
                    # This class is the only one that owns it, together with "elements"
                    if sys.getrefcount(el) == 3:
                        return el
            else:
                self.__cache[size] = []

            out = THREAD.array((size,), dtype=self.__dtype)
            self.__cache[size].append(out)

            return out


# Keep an ArrayCacheManager object for each data type
ARRAY_CACHE = {}


def free_gpu_cache():
    '''
    Free the arrays saved in the GPU cache.
    '''
    FFT_CACHE.clear()
    for c in ARRAY_CACHE.values():
        c.free_cache()


def get_array_cache(dtype):
    '''
    Given a data type, return the associated array cache.

    :param dtype: data type.
    :type dtype: numpy.dtype
    :returns: array cache.
    :rtype: ArrayCacheManager
    '''
    c = ARRAY_CACHE.get(dtype, None)
    if c is None:
        c = ArrayCacheManager(dtype)
        ARRAY_CACHE[dtype] = c
    return c

################################################################################



################################################################################
# Reduce functions #############################################################

def create_reduce_function(function, arr, default):
  '''
  Create a :class:`reikna.algorithms.Reduce` object with the given function
  implemented.

  :param function: function to parse.
  :type function: function
  :param arr: array to process.
  :type arr: reikna.cluda.Array
  :returns: object that applies the function on a given array.
  :rtype: numpy.float64
  '''
  snippet = reikna.cluda.Snippet.create(function)
  predicate = reikna.algorithms.Predicate(snippet, default)
  return reikna.algorithms.Reduce(arr, predicate).compile(THREAD)


def declare_reduce_function(function_proxy, default):
  '''
  Return a decorator to create a :class:`reikna.algorithms.Reduce` object
  to apply a reduction of an array to a single value.

  :param function_proxy: function to pass to :class:`reikna.cluda.Snippet`.
  :type function_proxy: function
  '''
  cache = {}

  def __wrapper(arr):
    callobj = cache.get(arr.shape, None)
    if callobj is None:
      callobj = create_reduce_function(function_proxy, arr, default)
      cache[arr.shape] = callobj
    result = THREAD.array((1,), dtype=arr.dtype)
    callobj(result, arr)
    return result.get().item()

  return __wrapper

################################################################################



################################################################################
# Operations over arrays #######################################################

def return_dtype(dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    cache_mgr = get_array_cache(dtype)

    def __wrapper(function):
        '''
        Internal wrapper.
        '''
        @functools.wraps(function)
        def __wrapper(arr, *args, **kwargs):
            '''
            Internal wrapper.
            '''
            gs, ls = get_sizes(len(arr))
            out = cache_mgr.get_array(len(arr))
            function(out, arr, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


RETURN_COMPLEX = return_dtype(types.cpu_complex)
RETURN_DOUBLE = return_dtype(types.cpu_type)
RETURN_BOOL = return_dtype(types.cpu_bool)


# Compile general GPU functions by element.
KERNEL_CODE = """
/** Definition of functions to execute in GPU arrays.
 */

/** Arange (only modifies real values)
 *
 * Reikna does not seem to handle very well complex numbers. Setting
 * "vmin" as a complex results in undefined behaviour some times.
 */
KERNEL void arange_complex( GLOBAL_MEM double *out, double vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx][0] = vmin + idx;
  out[idx][1] = 0.;
}

/// Arange
KERNEL void arange_int( GLOBAL_MEM int *out, int vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx;
}

/// Assign values
KERNEL void assign_double( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int offset )
{
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
}

/// Assign values
KERNEL void assign_bool( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in, int offset )
{
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
}

/// Exponential (complex)
KERNEL void exponential_complex( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  double v = in[idx];

  double d = exp(v[0]);

  out[idx][0] = d * cos(v[1]);
  out[idx][1] = d * sin(v[1]);
}

/// Exponential (double)
KERNEL void exponential_double( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  out[idx] = exp(x);
}

/// Linear interpolation
KERNEL void interpolate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int n, GLOBAL_MEM double *xp, GLOBAL_MEM double *yp )
{
  SIZE_T idx = get_global_id(0);

  double x = in[idx];

  for ( int i = 0; i < n; ++i ) {

    if ( x > xp[i] )
      continue;
    else {

      if ( x == xp[i] )
  out[idx] = yp[i];
      else
  out[idx] = (yp[i - 1]*(xp[i] - x) + yp[i]*(x - xp[i - 1])) / (xp[i] - xp[i - 1]);

      break;
    }
  }
}

/// Linspace (endpoints included)
KERNEL void linspace( GLOBAL_MEM double *out, double vmin, double vmax, int size )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx * (vmax - vmin) / (size - 1);
}

/// Logarithm
KERNEL void logarithm( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];
  out[idx] = log(x);
}

/// Greater or equal than
KERNEL void geq( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] >= v );
}

/// Less than (for arrays)
KERNEL void ale( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in1, GLOBAL_MEM double *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in1[idx] < in2[idx] );
}

/// Less than
KERNEL void le( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] < v );
}

/// Less or equal than
KERNEL void leq( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] <= v );
}

/// Logical and
KERNEL void logical_and( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] == in2[idx]);
}

/// Logical and
KERNEL void logical_or( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] || in2[idx]);
}

/// Create an array of ones
KERNEL void ones_bool( GLOBAL_MEM unsigned *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = true;
}

/// Create an array of ones
KERNEL void ones_double( GLOBAL_MEM double *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = 1.;
}

/// Take the real part of an array
KERNEL void real( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = in[idx][0];
}

/// Get elements from an array by indices
KERNEL void slice_from_integer( GLOBAL_MEM double *out, GLOBAL_MEM double *in, GLOBAL_MEM int *indices )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = in[indices[idx]];
}

/// Create an array filled with "false" till the given index
KERNEL void false_till( GLOBAL_MEM unsigned *out, int n )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx >= n);
}

/// Create an array filled with "true" till the given index
KERNEL void true_till( GLOBAL_MEM unsigned *out, int n )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx < n);
}

/// Create an array of zeros
KERNEL void zeros_bool( GLOBAL_MEM unsigned *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = false;
}

/// Create an array of zeros
KERNEL void zeros_double( GLOBAL_MEM double *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = 0.;
}
"""
FUNCS_BY_ELEMENT = THREAD.compile(KERNEL_CODE)

# These functions take an array of doubles and return another array of doubles
for function in ('exponential_complex',):
  setattr(FUNCS_BY_ELEMENT, function, RETURN_COMPLEX(
    getattr(FUNCS_BY_ELEMENT, function)))

# These functions take an array of doubles and return another array of doubles
for function in ('exponential_double', 'logarithm', 'real'):
    setattr(FUNCS_BY_ELEMENT, function, RETURN_DOUBLE(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions take an array of doubles as an input, and return an array of bool
for function in ('ale', 'geq', 'le', 'leq', 'logical_and', 'logical_or'):
    setattr(FUNCS_BY_ELEMENT, function, RETURN_BOOL(
        getattr(FUNCS_BY_ELEMENT, function)))


def creating_array_dtype(dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    cache_mgr = get_array_cache(dtype)

    def __wrapper(function):
        '''
        Internal wrapper.
        '''
        @functools.wraps(function)
        def __wrapper(size, *args, **kwargs):
            '''
            Internal wrapper.
            '''
            gs, ls = get_sizes(size)
            out = cache_mgr.get_array(size)
            function(out, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


CREATE_COMPLEX = creating_array_dtype(types.cpu_complex)
CREATE_DOUBLE = creating_array_dtype(types.cpu_type)
CREATE_INT = creating_array_dtype(types.cpu_int)
CREATE_BOOL = creating_array_dtype(types.cpu_bool)

# These functions create e new array of complex numbers
for function in ('arange_complex',):
  setattr(FUNCS_BY_ELEMENT, function, CREATE_COMPLEX(
    getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of doubles
for function in ('interpolate', 'linspace', 'ones_double', 'slice_from_integer', 'zeros_double'):
  setattr(FUNCS_BY_ELEMENT, function, CREATE_DOUBLE(
    getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of integers
for function in ('arange_int',):
  setattr(FUNCS_BY_ELEMENT, function, CREATE_INT(
    getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of bool
for function in ('false_till', 'true_till', 'ones_bool', 'zeros_bool'):
  setattr(FUNCS_BY_ELEMENT, function, CREATE_BOOL(
    getattr(FUNCS_BY_ELEMENT, function)))



def reikna_fft(a, inverse=False):
  '''
  Get the FFT to calculate the FFT of an array, keeping the compiled
  source in a cache.
  '''
  global FFT_CACHE

  # Compile the FFT
  cf = FFT_CACHE.get(a.shape, None)
  if cf is None:
    f = FFT(a)
    cf = f.compile(THREAD)
    FFT_CACHE[a.shape] = cf

  # Calculate the value
  output = get_array_cache(types.cpu_complex).get_array(len(a))

  cf(output, a, inverse=inverse)

  return output


# The declaration of functions starts here
rfuncs_amax = declare_reduce_function(
    lambda f, s: 'return ${f} > ${s} ? ${f} : ${s};', default=np.finfo(types.cpu_type).min)
rfuncs_amin = declare_reduce_function(
    lambda f, s: 'return ${f} < ${s} ? ${f} : ${s};', default=np.finfo(types.cpu_type).max)
rfuncs_rsum = declare_reduce_function(lambda f, s: 'return ${f} + ${s};', default=0)
rfuncs_count_nonzero = declare_reduce_function(
    lambda f, s: 'return ${f} + ${s};', default=0)


################################################################################



################################################################################
# Ristra functions #############################################################

def arange(n, dtype=types.cpu_int):
    if dtype == types.cpu_int:
        return FUNCS_BY_ELEMENT.arange_int(n, types.cpu_int(0))
    elif dtype == types.cpu_complex:
        return FUNCS_BY_ELEMENT.arange_complex(n, types.cpu_type(0))
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')



def ale(a1, a2):
    return FUNCS_BY_ELEMENT.ale(a1, a2)



def concatenate(arrays, maximum=None):

    maximum = maximum if maximum is not None else np.sum(
        np.fromiter(map(len, arrays), dtype=types.cpu_int))

    dtype = arrays[0].dtype

    if dtype == types.cpu_type:
        function = FUNCS_BY_ELEMENT.assign_double
    elif dtype == types.cpu_bool:
        function = FUNCS_BY_ELEMENT.assign_bool
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')

    out = get_array_cache(dtype).get_array(maximum)

    offset = types.cpu_int(0)
    for a in arrays:
        l = types.cpu_int(len(a))
        gs, ls = get_sizes(types.cpu_int(
            l if l + offset <= maximum else maximum - offset))
        function(out, a, types.cpu_int(
            offset), global_size=gs, local_size=ls)
        offset += l

    return out



def count_nonzero(a):
    return rfuncs_count_nonzero(a)



def allocate(a, copy=True, convert=True):
    if convert:
        if a.dtype != types.cpu_type:
            a = a.astype(types.cpu_type)
        return THREAD.to_device(a)
    # Is assumed to be a GPU-array
    if copy:
        return a.copy()
    else:
        return a



def empty(size, dtype=types.cpu_type):
    return get_array_cache(dtype).get_array(size)



def exp(a):
  if a.dtype == types.cpu_complex:
    return FUNCS_BY_ELEMENT.exponential_complex(a)
  elif a.dtype == types.cpu_type:
    return FUNCS_BY_ELEMENT.exponential_double(a)
  else:
    raise NotImplementedError(f'Not implemented for data type "{a.dtype}"')



def get(a):
  return a.get()



def false_till(N, n):
    return FUNCS_BY_ELEMENT.false_till(N, types.cpu_int(n))



def fft(a):
    return reikna_fft(a.astype(types.cpu_complex))



def fftconvolve(a, b, data):

    fa = fft(a)
    fb = fft(b)

    shift = fftshift(data)

    output = ifft(fa * shift * fb)

    return output * (data[1].get() - data[0].get())



def fftshift(a):
    n0 = count_nonzero(le(a, 0))
    nt = len(a)
    com = types.cpu_complex(+2.j * np.pi * n0 / nt)
    rng = arange(nt, dtype=types.cpu_complex)
    return exp(com * rng)



def geq(a, v):
    return FUNCS_BY_ELEMENT.geq(a, types.cpu_type(v))



def ifft(a):
    return reikna_fft(a, inverse=True)



def interpolate_linear(x, xp, yp):
    return FUNCS_BY_ELEMENT.interpolate(len(x), x, types.cpu_int(len(xp)), xp, yp)



def le(a, v):
    return FUNCS_BY_ELEMENT.le(a, types.cpu_type(v))



def leq(a, v):
    return FUNCS_BY_ELEMENT.leq(a, types.cpu_type(v))



def linspace(vmin, vmax, size):
    return FUNCS_BY_ELEMENT.linspace(size,
                                     types.cpu_type(vmin),
                                     types.cpu_type(vmax),
                                     types.cpu_int(size))



def log(a):
    return FUNCS_BY_ELEMENT.logarithm(a)



def log10(a):
    return FUNCS_BY_ELEMENT.logarithm(a)



def log2(a):
    return FUNCS_BY_ELEMENT.logarithm(a)



def logical_and(a, b):
    return FUNCS_BY_ELEMENT.logical_and(a, b)



def logical_or(a, b):
    return FUNCS_BY_ELEMENT.logical_or(a, b)



def max(a):
    return rfuncs_amax(a)



def meshgrid(*arrays):
    a = map(np.ndarray.flatten, np.meshgrid(*tuple(a.get() for a in arrays)))
    return tuple(map(THREAD.to_device, a))



def min(a):
    return rfuncs_amin(a)



def ones(n, dtype=types.cpu_type):
    if dtype == types.cpu_type:
        return FUNCS_BY_ELEMENT.ones_double(n)
    elif dtype == types.cpu_bool:
        return FUNCS_BY_ELEMENT.ones_bool(n)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')



def random_uniform(vmin, vmax, size):
    return THREAD.to_device(np.random.uniform(vmin, vmax, size))



def real(a):
    return FUNCS_BY_ELEMENT.real(a)



def shuffling_index(n):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return THREAD.to_device(indices)



def sum(a, *args):
  if len(args) == 0:
    if a.dtype == types.cpu_type:
      return rfuncs_rsum(a)
    else:
      raise NotImplementedError(f'Not implemented for data type {a.dtype}')
  else:
    r = a
    for a in args:
      r += a
    return r



def sum_inside(centers, edges, values=None):
  borders = meshgrid(*tuple(false_till(len(e) - 1, len(e) - 2)
                            for e in edges))

  gaps = tuple(map(lambda e: types.cpu_int(len(e)), edges))

  out = zeros(
      np.prod(np.fromiter((len(e) - 1 for e in edges), dtype=types.cpu_int)))

  gs, ls = get_sizes(len(out))

  gs = (len(centers[0]), gs)
  ls = (1, ls)

  if values is None:
    f = multi_par.sum_inside_bins(len(edges))
    f(out, *centers, *gaps, *edges, *borders, global_size=gs, local_size=ls)
  else:
    f = multi_par.sum_inside_bins_with_values(len(edges))
    f(out, *centers, *gaps, *edges, *borders,
      values, global_size=gs, local_size=ls)

  return out



def slice_from_boolean(a, valid):
    return THREAD.to_device(a.get()[valid.get().astype(types.cpu_real_bool)])



def slice_from_integer(a, indices):
    return FUNCS_BY_ELEMENT.slice_from_integer(len(indices), a, indices)



def true_till(N, n):
    return FUNCS_BY_ELEMENT.true_till(N, types.cpu_int(n))



def zeros(n, dtype=types.cpu_type):
  if dtype == types.cpu_type:
    return FUNCS_BY_ELEMENT.zeros_double(n)
  elif dtype == types.cpu_bool:
    return FUNCS_BY_ELEMENT.zeros_bool(n)
  else:
    raise NotImplementedError(f'Not implemented for data type "{dtype
