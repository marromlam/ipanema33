#include "random.h"

WITHIN_KERNEL
float fract(float x)
{
    return 0;//fmin( x - floor(x), 0x1.fffffep-1f );
}


// core or Romero's way to create random numbers in openCL
WITHIN_KERNEL
int rng_gin(int seed, int cycles)
{
	for (int i=1; i<=cycles; i++)
  {
    seed = ( seed * rngA )%rngM;
    seed = ( seed - rngM * floor ( seed * rngN ) );
  }
  return seed;
}


// Basic uniform generators
WITHIN_KERNEL
int rng_uniform_int(void * seed, int cycles)
{
  return rng_gin(*(int*)seed, cycles);
}



WITHIN_KERNEL
float rng_uniform_float(void * seed, int cycles)
{
  return (float) rng_gin(*(int*)seed, cycles)/rngM;
}



WITHIN_KERNEL
ftype rng_uniform(void * seed, int cycles)
{
  #ifdef CUDA
    return curand_uniform((curandState*)seed);
  #else
    #if USE_DOUBLE
      return convert_double( rng_gin(*(int*)seed, cycles) )/rngM;
    #else
      return convert_float( rng_gin(*(int*)seed, cycles) )/rngM;
    #endif
  #endif
}

// other PDF uniform generators

// Box-Muller for gaussian random numbers
WITHIN_KERNEL
ftype rngNormal(ftype mu, ftype sigma, void * seed, int cycles)
{
  ftype x = rng_uniform(seed, cycles); //WARNING
  int _seed = *(int*)seed + x;
  ftype y = rng_uniform(&_seed, cycles);
  //printf("%f, %f\n", x,y);
  ftype z = sqrt( -2.0*log(x) ) * cos( 2.0*M_PI*y );
  return mu + z*sigma;
}



WITHIN_KERNEL
ftype rngLogNormal(ftype mu, ftype sigma, void * seed, int cycles)
{
  #ifdef CUDA
    return curand_log_normal((curandState*)seed, mu, sigma);
  #else
    ftype phi = sqrt(sigma*sigma + mu*mu);
    ftype mu_ = log(mu*mu/phi);
    ftype sigma_ = sqrt(log(phi*phi/(mu*mu)));
    return exp( rngNormal(mu_, sigma_, seed, cycles) );
  #endif
}
