#ifndef _CORE_H_
#define _CORE_H_

// Define machine constants
#define SQRT_2PI_INV 0.3989422804
#define EUL 5.772156649015328606065e-1
#define MAXFAC 31
#define MAXNUM 1.79769313486231570815E308     // 2**1024*(1-MACHEP)
#define MAXLOG 8.8029691931113054295988E1     // log(2**127)
#define MACHEP 1.38777878078144567553E-17     // 2**-56
#define DBLEPS 2.2204460492503131e-16
// MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
// MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
#define BIG 4.503599627370496e15
#define BIGINV 2.22044604925031308085e-16

// More math constants
#define M_SQRTPI_2 1.2533141373155001 // sqrt(pi/2)
#define M_SQRTPIHALF 0.8862269254527580136490837416705725913990 // sqrt(pi)/2
#define M_SQRT2PI 2.5066282746310005024157652848110 // sqrt(2*pi)

//#define M_SQRT2 1.4142135623730951

// these ones are for faddeva
#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29

#define Inf (1./0.)
#define NaN (0./0.)

#if USE_DOUBLE
  #define ftype double
  #define ctype double2
#else
  #define ftype float
  #define ctype float2
#endif

/*
-------------------------------------------------------------------------------
openCL standard math functions
-------------------------------------------------------------------------------

acos      acosh       acospi    asin
asinh	    asinpi      atan      atan2
atanh	    atanpi	    atan2pi	  cbrt
ceil	    copysign	  cos	      cosh
cospi	    erfc	      erf	      exp
exp2	    exp10	      expm1	    fabs
fdim	    floor	      fma	      fmax
fmin	    fmod	      fract	    frexp
hypot	    ilogb	      ldexp	    lgamma
lgamma_r  log	        log2	    log10
log1p	    logb	      mad	      maxmag
minmag	  modf	      nan	      nextafter
pow	      pown	      powr	    remainder
remquo	  rint	      rootn	    round
rsqrt	    sin	        sincos	  sinh
sinpi	    sqrt	      tan	      tanh
tanpi	    tgamma	    trunc

see www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/mathFunctions.html
*/


WITHIN_KERNEL
ftype rpow(const ftype x, const ftype n);



WITHIN_KERNEL
ftype sqr(const ftype x);



#endif //_CORE_H_
