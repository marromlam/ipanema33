#define rngA 16807      //ie 7**5
#define rngM 2147483647 //ie 2**31-1
#define rngN 0.00000000046566128752457969241057508271679984532147638747537340385623 //ie 1/(2**31-1)

#include "core.hpp"


#ifdef CUDA
  #include <curand.h>
  #include <curand_kernel.h>
  //curandState state;
  //curand_init((unsigned long long)clock(), evt, 0, &state);
#endif




WITHIN_KERNEL
float fract(float x);

// WITHIN_KERNEL
// int rng_core(int seed, int cycles);
//
// WITHIN_KERNEL
// int rng_uniform_int(int seed, int cycles);
//
// WITHIN_KERNEL
// ftype rng_uniform(int seed, int cycles);
//
// WITHIN_KERNEL
// float rng_uniform_float(int seed, int cycles);
//




KERNEL
void rngin_uniform_int(GLOBAL_MEM int *out, int seed, int cycles);



KERNEL
void rngin_uniform(GLOBAL_MEM double *out, int seed, int cycles);


KERNEL
void rngin_uniform_float(GLOBAL_MEM double *out, int seed, int cycles);
