#include "core.h"

#if USE_DOUBLE
  #ifndef CUDA
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
    WITHIN_KERNEL
    void
    atomicAdd(volatile __global double *addr, double val)
    {
      union {
        long u;
        double f;
      } next, expected, current;
      current.f = *addr;
      do {
        expected.f = current.f;
        next.f = expected.f + val;
        current.u = atomic_cmpxchg( (volatile __global long *) addr, expected.u, next.u);
      } while( current.u != expected.u );
    }
  #endif
#endif



#ifdef CUDA
  WITHIN_KERNEL
  ftype fract(const ftype x)
  {
    return x - floorf(x);
  }
#else
  WITHIN_KERNEL
  ftype fract(const ftype x)
  {
    return x - floor(x);
  }
#endif



WITHIN_KERNEL
ftype rpow(const ftype x, const ftype n)
{
  return pow((ftype) x, (ftype) n);
}



WITHIN_KERNEL
ftype sqr(const ftype x)
{
  return x*x;
}
