################################################################################
#                                                                              #
#                        BACKEND SELECTOR & INITIALIZE                         #
#                                                                              #
################################################################################


__author__ = 'Marcos Romero Lamas'
__email__ = 'marromlam@gmail.com'
__all__ = ['PYTHON', 'OPENCL', 'CUDA', 'fetch_devices', 'initialize', 'deinitialize',
           'get_sizes', 'initialize_device', 'ristra', 'compile']


# Modulo {{{

import os
import builtins
from reikna import cluda
import atexit
import numpy as np
import math
from lib99ocl import LIB99OCL

# }}}


# Backends {{{
#    This is where operations take place

# Three backends:
PYTHON = 'python' # python: standard backend (no compiled code)
OPENCL = 'opencl' # opencl: single-core, multi-core, gpu
CUDA = 'cuda'     # nvidia: NVIDIA-gpu

builtins.BACKEND = None
builtins.DEVICE = None
builtins.CONTEXT = None
builtins.THREAD = None
builtins.REAL = 'float'

# }}}


# Array allocation {{{
#    This is where arrays will be allocated, and this depends of course on then
#    selected backend.

# Initial ALLOCATION
builtins.ALLOCATION = None
MAX_LOCAL_SIZE = 128

class manipulate_array(type):
  def __getattr__(cls, name):
    return getattr(builtins.ALLOCATION, name)

class ristra(metaclass=manipulate_array):
  pass

# }}}


# Compile function {{{

def compile(template:str, render_args:list=None, render_kwds:dict=None,
            fast_math:bool=True, compiler_options:list=None, keep:bool=False):
    """
    Compiles a template against the `ipanema.BACKEND`. It includes all the
    functions defined in lib99ocl library. It returns a reikna.Program object
    which has all the defined kernels as callable methods.

    Parameters
    ----------
    template:str
    Code string to be compiled.
    render_args:list

    """
    # add lib99ocl to the list of included directories
    if not compiler_options:
        compiler_options = []
    compiler_options += [f"-I {LIB99OCL}"]

    return THREAD.compile(template_src=template, render_args=render_args, 
                          render_kwds=render_kwds, fast_math=fast_math,
                          compiler_options=compiler_options, keep=keep)

# }}}


# Device functions {{{
#    Here several functions are declared in order to find and initialize the
#    selected from the avaliable devices

def fetch_devices():
  API = cluda.ocl_api()
  platforms = API.get_platforms()
  all_devices = [(p, d) for p in platforms for d in p.get_devices()]
  print('Host {} has the following backends:'.format(os.uname()[1]))
  print(f"idx   vendor          device ")
  for i, pla_dev  in enumerate(all_devices):
    platform, device = pla_dev
    print(f"[{i+1}]   <{platform.name}>   {device.name} ")
  print('Please select one [device]')


def initialize_device(device=None, verbose=False):
  if not device:
    fetch_devices()
    return
  else:
    # Get api accordingly to backend
    if builtins.BACKEND == CUDA:
      API = cluda.cuda_api()
    elif builtins.BACKEND == OPENCL:
      API = cluda.ocl_api()

    # Get all available devices and then choose device
    platforms = API.get_platforms()
    all_devices = [(p, d) for p in platforms for d in p.get_devices()]
    platform, device = all_devices[device-1]

    builtins.DEVICE = device

    # Create the context and THREAD
    if builtins.BACKEND == CUDA:
      builtins.CONTEXT = builtins.DEVICE.make_context()
      def clear_cuda_context():
          from pycuda.tools import clear_context_caches
          builtins.CONTEXT.pop()
          clear_context_caches()
      atexit.register(clear_cuda_context)
    else:
      import pyopencl
      builtins.CONTEXT = pyopencl.Context([DEVICE])

    builtins.THREAD = API.Thread(builtins.CONTEXT)
    if verbose: print(f'Selected device <{platform.name}> : {device.name}')

# }}}


# Get sizes function {{{

def get_sizes(size, BLOCK_SIZE=MAX_LOCAL_SIZE):
  a = size % BLOCK_SIZE
  if a == 0:
    gs, ls = size, BLOCK_SIZE
  elif size < BLOCK_SIZE:
    gs, ls = size, 1
  else:
    a = np.arange(1, min(BLOCK_SIZE, math.ceil(math.sqrt(size))))
    a = a[size % a == 0]
    ls = int(a[np.argmin(np.abs(a - BLOCK_SIZE))])
    gs = size
  return int(gs), int(ls)
# def get_sizes(size, BLOCK_SIZE=256):
#     '''
#     i need to check if this worls for 3d size and 3d block
#     '''
#     a = size % BLOCK_SIZE
#     if a == 0:
#       gs, ls = size, BLOCK_SIZE
#     elif size < BLOCK_SIZE:
#       gs, ls = size, 1
#     else:
#       a = np.ceil(size/BLOCK_SIZE)
#       gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
#     return int(gs), int(ls)

# }}}


# Initialization {{{
#    This function should be called at the very beginning of the code. It sets
#    the BACKEND, DEVICE, CONTEXT, THREAD and ALLOCATION. So it puts in place
#    all the needed ingredients to ipanema work

def initialize(backend=PYTHON, device=None, verbose=False, real='double'):
  if builtins.BACKEND is not None and backend != builtins.BACKEND:
    print(f'Unable to set backend to "{backend}". It is already "{BACKEND}".')
    return
  elif backend == builtins.BACKEND:
    print(f'Already set "{backend}" as ({BACKEND})')
    return
  elif backend == PYTHON:
    pass
  elif builtins.BACKEND != "python" and (device is None or device == 0):
    builtins.BACKEND = None
    initialize_device(device, verbose=verbose)
    return

  builtins.BACKEND = backend
  builtins.REAL = real

  if builtins.BACKEND == PYTHON:
    if verbose: print(f'Using backend "{builtins.BACKEND}"')
    from . import python
    builtins.ALLOCATION = python
  elif builtins.BACKEND == CUDA or builtins.BACKEND == OPENCL:
    if verbose: print(f'Using backend "{builtins.BACKEND}"')
    initialize_device(device, verbose=verbose)
    from . import device
    builtins.ALLOCATION = device
  else:
    raise ValueError(f'Unknown backend "{BACKEND}"')


def deinitialize():
  builtins.BACKEND = None
  builtins.DEVICE = None
  builtins.CONTEXT = None
  builtins.THREAD = None
  builtins.ALLOCATION = None
  print('this function is a placeholder...')

# }}}


# vim:foldmethod=marker
