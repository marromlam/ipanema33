################################################################################
#                                                                              #
#                                   TYPES                                      #
#                                                                              #
################################################################################

import ctypes
import numpy



# Types for numpy.ndarray objects
cpu_type      = numpy.float64     # double
cpu_complex   = numpy.complex128  # complex double
cpu_int       = numpy.int32       # int
cpu_bool      = numpy.uint32
cpu_real_bool = numpy.bool        # bool (not allowed in PyOpenCL)

# Types to handle with ctypes
c_int         = ctypes.c_int  # int
c_double      = ctypes.c_double  # double
c_double_p    = ctypes.POINTER(c_double)  # double*
