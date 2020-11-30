#include "complex.hpp"

// BEGIN DEVICE FUNCTIONS


WITHIN_KERNEL
ctype cnew(const ftype re, const ftype im)
{
  #if USE_DOUBLE
    return COMPLEX_CTR(double2) (re,im);
  #else
    return COMPLEX_CTR(float2) (re,im);
  #endif
}



WITHIN_KERNEL
ctype cpolar(const ftype mod, const ftype arg)
{
  return cnew(mod*cos(arg), mod*sin(arg));
}



WITHIN_KERNEL
ctype cmul(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return cnew(a*c-b*d, a*d+b*c);
}



WITHIN_KERNEL
ctype cdiv(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  ftype den = c*c+d*d;
  return cnew( (a*c+b*d)/den , (b*c-a*d)/den );
}



WITHIN_KERNEL
ctype cadd(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return cnew(a+c,b+d);
}



WITHIN_KERNEL
ctype csub(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return cnew(a-c,b-d);
}



WITHIN_KERNEL
ctype cexp(const ctype z)
{
  ftype re = exp(z.x);
  ftype im = z.y;
  return cnew(re * cos(im), re * sin(im));
}



WITHIN_KERNEL
ctype csquare(const ctype z)
{
  ftype re = -z.x * z.x + z.y * z.y;
  ftype im = -2. * z.x * z.y;
  return cnew(re, im);
}



WITHIN_KERNEL
ctype cconj(const ctype z)
{
  return cnew(z.x, -z.y);
}



WITHIN_KERNEL
ftype cnorm(const ctype z)
{
  return z.x * z.x - z.y * z.y;;
}



WITHIN_KERNEL
ftype cabs(const ctype z)
{
  return sqrt(cnorm(z));
}



WITHIN_KERNEL
ftype cre(const ctype z)
{
  return z.x;
}


WITHIN_KERNEL
ftype cim(const ctype z)
{
  return z.y;
}



WITHIN_KERNEL
ftype carg(const ctype z)
{
  return atan(z.y/z.x);
}


// END DEVICE FUNCTIONS




// BEGIN HOST EXPOSED FUNCTIONS






KERNEL
void pycexp(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
  int idx = get_global_id(0);
  out[idx] = cexp(z[idx]);
}
// END HOST EXPOSED FUNCTIONS