#include "random.h"


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
int rng_uniform_int(void * seed)
{
  return rng_gin(*(int*)seed, RNG_CYCLES);
}


WITHIN_KERNEL
float rng_uniform_float(void * seed)
{
  return (float) rng_gin(*(int*)seed, RNG_CYCLES)/rngM;
}


WITHIN_KERNEL
ftype rng_uniform(void * seed)
{
  #ifdef CUDA
    return curand_uniform((curandState*)seed);
  #else
    #if USE_DOUBLE
      return convert_double( rng_gin(*(int*)seed, RNG_CYCLES) )/rngM;
    #else
      return convert_float( rng_gin(*(int*)seed, RNG_CYCLES) )/rngM;
    #endif
  #endif
}

// other PDF uniform generators

// Box-Muller for gaussian random numbers
WITHIN_KERNEL
ftype rng_normal(const ftype mu, const ftype sigma, void * seed)
{
  const ftype x = rng_uniform(seed);
  int _seed = *(int*)seed + x;
  const ftype y = rng_uniform(&_seed);
  const ftype z = sqrt( -2.0*log(x) ) * cos( 2.0*M_PI*y );
  return mu + z*sigma;
}



WITHIN_KERNEL
ftype rng_log_normal(const ftype mu, const ftype sigma, void * seed)
{
  #ifdef CUDA
    return curand_log_normal((curandState*)seed, mu, sigma);
  #else
    const ftype __phi = sqrt(sigma*sigma + mu*mu);
    const ftype __mu = log(mu*mu/__phi);
    const ftype __sigma = sqrt(log(__phi*__phi/(mu*mu)));
    return exp( rng_normal(__mu, __sigma, seed) );
  #endif
}
