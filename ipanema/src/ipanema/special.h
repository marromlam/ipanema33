#ifndef _SPECIAL_H_
#define _SPECIAL_H_
//#include "complex.hpp"
#include "core.h"
#include "details/bessel.c"
#include "details/gamma.c"

#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  FACTORIAL                                                                 //
////////////////////////////////////////////////////////////////////////////////



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
int nfactorial(const int n);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype factorial(const int n);



WITHIN_KERNEL
ftype binom(const int n, const int k);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE ERROR FUNCTION FAMILY                                                  //
////////////////////////////////////////////////////////////////////////////////
#include "details/cerf.c"



WITHIN_KERNEL
ftype erfcx(const ftype x);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  POLYNOMIALS and friends                                                   //
////////////////////////////////////////////////////////////////////////////////


/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype lpmv(const int l, const int m, const ftype cos_theta);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype sph_harm(const int l, const int m, const ftype cos_theta, const ftype phi);




WITHIN_KERNEL
ftype polyBernstein(const ftype x, const ftype *c, const int n)
{
  ftype ans = 0.0;
  for (int k=0; k<n; k++)
  {
    ans += binom(n, k) * rpow(x, k) * rpow(1-x, n-k);
  }
  return ans;
}


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  THE GAMMA FAMILY                                                          //
////////////////////////////////////////////////////////////////////////////////

/*
jv(v, z)
Bessel function of the first kind of real order and complex argument.
jve(v, z)
Exponentially scaled Bessel function of order v.
yn(n, x)
Bessel function of the second kind of integer order and real argument.
yv(v, z)
Bessel function of the second kind of real order and complex argument.
yve(v, z)
Exponentially scaled Bessel function of the second kind of real order.
kn(n, x)
Modified Bessel function of the second kind of integer order n
kv(v, z)
Modified Bessel function of the second kind of real order v
kve(v, z)
Exponentially scaled modified Bessel function of the second kind.
iv(v, z)
Modified Bessel function of the first kind of real order.
ive(v, z)
Exponentially scaled modified Bessel function of the first kind
hankel1(v, z)
Hankel function of the first kind
hankel1e(v, z)
Exponentially scaled Hankel function of the first kind
hankel2(v, z)
Hankel function of the second kind
hankel2e(v, z)
Exponentially scaled Hankel function of the second kind
*/


/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rgammaln(const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rgamma(const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rgammainc(const ftype a, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rgammaincc(const ftype a, const ftype x);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  THE BESSEL FAMILY                                                         //
////////////////////////////////////////////////////////////////////////////////



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rjn(const int n, const ftype x);



/**
 * Bessel Jv
 *
 * Bessel 1st of real order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rjv(const ftype n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype ryn(const int n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype ryv(const ftype n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rin(const int n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype riv(const ftype n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rin(const int n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype riv(const ftype n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rkn(const int n, const ftype x);



/**
 * Bessel Jn
 *
 * Bessel 1st of integer order
 *
 * @param n Order.
 * @param x Point
 * @return Bessel J function of order n at x.
 */
WITHIN_KERNEL
ftype rkv(const ftype n, const ftype x);

#endif // _SPECIAL_H_
