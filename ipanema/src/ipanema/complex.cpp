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
  return z.x * z.x + z.y * z.y;
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


WITHIN_KERNEL
ctype clog(const ctype z)
{
  return cnew( log(cabs(z)) , atan2( cim(z), cre(z) ) ); 
}






// special functions


WITHIN_KERNEL
ctype __core_cigamc(ftype a, ctype x)
{
    const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
    const ftype BIG = 4.503599627370496e15;
    const ftype BIGINV = 2.22044604925031308085e-16;

    // Compute  x**a * exp(-x) / gamma(a)  
    ctype ax = cmul( cnew(a,0) , clog(x) );
    ax = csub( ax, x);
    ax = csub( ax, cnew(lgamma(a),0) );
    if (cabs(ax) < -MAXLOG) return cnew(0.,0.); // underflow
    ax = cexp(ax);

    // Continued fraction implementation
    ctype y = cnew(1.-a, 0);
    ctype z = cadd(x,cadd(y, cnew(1.,0)));
    ctype c = cnew(0.,0.);
    ctype pkm2 = cnew(1.,0.);
    ctype qkm2 = x;
    ctype pkm1 = cadd(x, cnew(1.,0));
    ctype qkm1 = cmul(z, x);
    ctype ans = cdiv(pkm1, qkm1);
    ctype yc, pk, qk, r;
    ftype t;

    do {
        c = cadd(c ,cnew(1.,0.));
        y = cadd(y ,cnew(1.,0.));
        z = cadd(z ,cnew(2.,0.));
        yc = cmul(y, c);
        pk = csub( cmul(pkm1, z) , cmul(pkm2, yc) );
        qk = csub( cmul(qkm1, z) , cmul(qkm2, yc) );
        if (cabs(qk) != 0) {
            r = cdiv(pk, qk);
            t = cabs( cdiv(csub(ans, r),r) );
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (cabs(pk) > BIG) {
            pkm2 = cmul(pkm2 ,cnew(BIGINV,0));
            pkm1 = cmul(pkm1 ,cnew(BIGINV,0));
            qkm2 = cmul(qkm2 ,cnew(BIGINV,0));
            qkm1 = cmul(qkm1 ,cnew(BIGINV,0));
        }
    } while( t > MACHEP );

    return cmul(ans, ax);
}



WITHIN_KERNEL
ctype __core_cigam(ftype a, ctype x)
{
    const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ctype ax = cmul( cnew(a,0) , clog(x) );
    ax = csub( ax, x);
    ax = csub( ax, cnew(lgamma(a),0) );

    //printf("log(x)= %+f %+f i\\n", cre(clog(x)), cim(clog(x)) );
    //printf("ax= %+f %+f i\\n", cre(ax), cim(ax) );
    if (cabs(ax) < -MAXLOG) return cnew(0.,0.); // underflow
    ax = cexp(ax);

    /* power series */
    ctype r = cnew(a,0);
    ctype c = cnew(1.0,0);
    ctype ans = cnew(1.0,0);

    do {
        r = cadd(r, cnew(1.,0.) );
        c = cmul( c , cdiv(x, r) );
        ans = cadd(ans, c);
        //printf(" * %f \\n", cnorm(cdiv(c,ans)) );
    } while (cnorm(cdiv(c,ans)) > MACHEP);

    return cmul(ans, cdiv(ax,cnew(a,0.)) );
}



WITHIN_KERNEL
ctype cgammaincc(ftype a, ctype z)
{
ctype ans = cnew(0,0);
    if ((cabs(z) <= 0.0) || (a <= 0)) return cnew(1.,0.);
    if ((cabs(z) <  1.0) || (cabs(z) <  a)) return csub( cnew(1.,0.) , __core_cigam(a, z));
    return  __core_cigamc(a, z);    
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
