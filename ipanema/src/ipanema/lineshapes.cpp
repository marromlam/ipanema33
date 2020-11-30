#include "core.hpp"
#define SQRTPI_2 1.2533141373155001
#define SQRT2 1.4142135623730951

KERNEL
void Amoroso( GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *in, ftype a, ftype theta, ftype alpha, ftype beta )
{

  SIZE_T idx = get_global_id(0);

  ftype x = in[idx];

  ftype d = (x - a) / theta;

  out[idx] = pow(d, alpha * beta - 1) * exp(- pow(d, beta));
}


KERNEL 
void CrystalBall( GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *in, ftype c, ftype s, ftype a, ftype n )
{
  SIZE_T idx = get_global_id(0);
  ftype x = in[idx];

  ftype t = ( a < 0 ? -1 : +1 ) * ( x - c ) / s;

  ftype aa = fabs(a);

  if ( t >= -aa )
    out[idx] = exp(-0.5 * t * t);
  else
  {
    ftype A = pow(n / aa, n) * exp(-0.5 * aa * aa);
    ftype B = n / aa - aa;

    out[idx] = A / pow(B - t, n);
  }
}




KERNEL 
void Normal( GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *in, ftype c, ftype s )
{
  SIZE_T idx = get_global_id(0);
  ftype x = in[idx];

  ftype s2 = s * s;
  ftype d  = (x - c);

  out[idx] = exp(-d * d / (2 * s2));
}


KERNEL 
void Poly( GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *in, int n, GLOBAL_MEM ftype *p )
{

  SIZE_T idx  = get_global_id(0);
  ftype x = in[idx];

  if ( n == 0 )
  {
    out[idx] = 1.;
    return;
  }

  ftype o = x * p[n - 1];
  for ( int i = 1; i < n; ++i )
    o = x * (o + p[n - i - 1]);

  out[idx] = o + 1.;
}


KERNEL 
void PowerLaw( GLOBAL_MEM ftype *out, GLOBAL_MEM ftype *in, ftype c, ftype n )
{
  SIZE_T idx  = get_global_id(0);
  ftype x = in[idx];

  out[idx] = 1. / pow(fabs(x - c), n);
}