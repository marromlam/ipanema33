################################################################################
#                                                                              #
#                                   TYPES                                      #
#                                                                              #
################################################################################

import ctypes
import numpy as np
from builtins import REAL


# Types for numpy.ndarray objects
cpu_real      = np.float64 if REAL == 'double' else np.float32
cpu_complex   = np.complex128 if REAL == 'double' else np.complex64
cpu_int       = np.int32
cpu_bool      = np.uint32
cpu_real_bool = np.bool


# Types to handle with ctypes
c_int         = ctypes.c_int
c_double      = ctypes.c_double
c_double_p    = ctypes.POINTER(c_double)
