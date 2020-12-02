################################################################################
#                                                                              #
#                        BACKEND SELECTOR & INITIALIZE                         #
#                                                                              #
################################################################################

import os
import builtins
from reikna import cluda
import atexit
import numpy as np
import math


# Backends ---------------------------------------------------------------------
#    This is where operations take place

# Three backends:
PYTHON = 'python' # python: standard backend (no compiled code)
OPENCL = 'opencl' # opencl: single-core, multi-core, gpu
CUDA = 'cuda'     # nvidia: NVIDIA-gpu

# Initial BACKEND
builtins.BACKEND = None

# Device variables
builtins.DEVICE = None
builtins.CONTEXT = None
builtins.THREAD = None


# Array allocation -------------------------------------------------------------
#    This is where arrays will be allocated, and this depends of course on then
#    selected backend.

# Initial ALLOCATION
builtins.ALLOCATION = None
MAX_LOCAL_SIZE = 100

class manipulate_array(type):
  def __getattr__(cls, name):
    return getattr(builtins.ALLOCATION, name)

class ristra(metaclass=manipulate_array):
  pass



# Device functions -------------------------------------------------------------
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



def initialize_device(device = None, verbose=False):
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



# def get_sizes(size):
#   a = size % MAX_LOCAL_SIZE
#   if a == 0:
#     gs, ls = size, MAX_LOCAL_SIZE
#   elif size < MAX_LOCAL_SIZE:
#     gs, ls = size, 1
#   else:
#     a = np.arange(1, min(MAX_LOCAL_SIZE, math.ceil(math.sqrt(size))))
#     a = a[size % a == 0]
#     ls = int(a[np.argmin(np.abs(a - MAX_LOCAL_SIZE))])
#     gs = size
#   return int(gs), int(ls)
def get_sizes(size, BLOCK_SIZE=256):
    '''
    i need to check if this worls for 3d size and 3d block
    '''
    a = size % BLOCK_SIZE
    if a == 0:
      gs, ls = size, BLOCK_SIZE
    elif size < BLOCK_SIZE:
      gs, ls = size, 1
    else:
      a = np.ceil(size/BLOCK_SIZE)
      gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
    return int(gs), int(ls)


# Initialization ---------------------------------------------------------------
#    This function should be called at the very beginning of the code. It sets
#    the BACKEND, DEVICE, CONTEXT, THREAD and ALLOCATION. So it puts in place
#    all the needed ingredients to ipanema work

def initialize(backend=PYTHON, device=None, verbose=False):
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
