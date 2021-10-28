#include "core.h"
#include "special.h"
#include "complex.h"

#define I C(0.,1.)

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// BASICS                                                                     //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype C(const ftype re, const ftype im)
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
  return C(mod*cos(arg), mod*sin(arg));
}



WITHIN_KERNEL
ctype cmul(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return C(a*c-b*d, a*d+b*c);
}



WITHIN_KERNEL
ctype cdiv(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  ftype den = c*c+d*d;
  return C( (a*c+b*d)/den , (b*c-a*d)/den );
}



WITHIN_KERNEL
ctype cadd(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return C(a+c,b+d);
}



WITHIN_KERNEL
ctype csub(const ctype z1, const ctype z2)
{
  ftype a = z1.x;
  ftype b = z1.y;
  ftype c = z2.x;
  ftype d = z2.y;
  return C(a-c,b-d);
}



WITHIN_KERNEL
ctype cexp(const ctype z)
{
  ftype re = exp(z.x);
  ftype im = z.y;
  return C(re * cos(im), re * sin(im));
}



WITHIN_KERNEL
ctype csquare(const ctype z)
{
  ftype re = -z.x * z.x + z.y * z.y;
  ftype im = -2. * z.x * z.y;
  return C(re, im);
}



WITHIN_KERNEL
ctype cconj(const ctype z)
{
  return C(z.x, -z.y);
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
  return C( log(cabs(z)) , atan2( cim(z), cre(z) ) );
}



WITHIN_KERNEL
ctype cpow(const ctype w, const ctype z)
{
  return cexp( cmul(z, clog(w) ));
}



WITHIN_KERNEL
ctype csqrt(const ctype w, const ctype z)
{
  return cexp( cmul(z, clog(w) ));
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// POLYNOMIALS and friends                                                    //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype csph_harm(const int m, const int l, const ftype cosT, const ftype phi)
{
  ftype ans = lpmv(m, l, cosT);
  ans *= sqrt( ((2*l+1)*factorial(l-m)) / (4*M_PI*factorial(l+m)) );
  return C(ans*cos(m*phi), ans*sin(m*phi));
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE COMPLEX GAMMA FAMILY                                                   //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype cgammaincc(ftype a, ctype z)
{
ctype ans = C(0,0);
    if ((cabs(z) <= 0.0) || (a <= 0)) return C(1.,0.);
    if ((cabs(z) <  1.0) || (cabs(z) <  a)) return csub( C(1.,0.) , __core_cigam(a, z));
    return  __core_cigamc(a, z);
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE COMPLEX BESSEL FAMILY                                                  //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype cjv(const ftype n, const ctype x)
{
  #ifdef CUDA
    /* printf("not yet implemented\n"); */
    return C(0.,0.);
  #else
    return C(0.,0.);
  #endif
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE FADDEVA FAMILY                                                         //
////////////////////////////////////////////////////////////////////////////////

// Most of the following code was taken from:
//    https://jugit.fz-juelich.de/mlz/libcerf.git
// and adapted to openCL standard. Some simplifications/prettifications were
// done.



// WITHIN_KERNEL
// ctype cwofz(ctype z)
// {
//   int faddeeva_algorithm;
//   int faddeeva_nofterms;
// 
//   const double expa2n2[] = {
//   7.64405281671221563e-01, 3.41424527166548425e-01, 8.91072646929412548e-02,
//   1.35887299055460086e-02, 1.21085455253437481e-03, 6.30452613933449404e-05,
//   1.91805156577114683e-06, 3.40969447714832381e-08, 3.54175089099469393e-10,
//   2.14965079583260682e-12, 7.62368911833724354e-15, 1.57982797110681093e-17,
//   1.91294189103582677e-20, 1.35344656764205340e-23, 5.59535712428588720e-27,
//   1.35164257972401769e-30, 1.90784582843501167e-34, 1.57351920291442930e-38,
//   7.58312432328032845e-43, 2.13536275438697082e-47, 3.51352063787195769e-52,
//   3.37800830266396920e-57, 1.89769439468301000e-62, 6.22929926072668851e-68,
//   1.19481172006938722e-73, 1.33908181133005953e-79, 8.76924303483223939e-86,
//   3.35555576166254986e-92, 7.50264110688173024e-99, 9.80192200745410268e-106,
//   7.48265412822268959e-113, 3.33770122566809425e-120, 8.69934598159861140e-128,
//   1.32486951484088852e-135, 1.17898144201315253e-143, 6.13039120236180012e-152,
//   1.86258785950822098e-160, 3.30668408201432783e-169, 3.43017280887946235e-178,
//   2.07915397775808219e-187, 7.36384545323984966e-197, 1.52394760394085741e-206,
//   1.84281935046532100e-216, 1.30209553802992923e-226, 5.37588903521080531e-237,
//   1.29689584599763145e-247, 1.82813078022866562e-258, 1.50576355348684241e-269,
//   7.24692320799294194e-281, 2.03797051314726829e-292, 3.34880215927873807e-304,
//   0.0 // underflow (also prevents reads past array end, below)
//   };
// 
//   if (cre(z) == 0.0)
//   {
//     // Purely imaginary input, purely real output.
//     // However, use cre(z) to give correct sign of 0 in cim(w).
//     return C(erfcx(cim(z)), cre(z));
//   }
//   if (cim(z) == 0)
//   {
//     // Purely real input, complex output.
//     return C(exp(-sqr(cre(z))),  im_w_of_x(cre(z)));
//   }
// 
//   const ftype a = 0.518321480430085929872; // pi / sqrt(-log(eps*0.5))
//   const ftype c = 0.329973702884629072537; // (2/pi) * a;
//   const ftype a2 = 0.268657157075235951582; // a^2
// 
//   const ftype x = fabs(cre(z));
//   const ftype y = cim(z);
//   const ftype ya = fabs(y);
// 
//   ctype ret = C(0.,0.); // return value
// 
//   ftype sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
// 
//   if (ya > 7 || (x > 6  // continued fraction is faster
//                  /* As pointed out by M. Zaghloul, the continued
//                     fraction seems to give a large relative error in
//                     Re w(z) for |x| ~ 6 and small |y|, so use
//                     algorithm 816 in this region: */
//                  && (ya > 0.1 || (x > 8 && ya > 1e-10) || x > 28))) {
// 
//       faddeeva_algorithm = 100;
// 
//       /* Poppe & Wijers suggest using a number of terms
//          nu = 3 + 1442 / (26*rho + 77)
//          where rho = sqrt((x/x0)^2 + (y/y0)^2) where x0=6.3, y0=4.4.
//          (They only use this expansion for rho >= 1, but rho a little less
//          than 1 seems okay too.)
//          Instead, I did my own fit to a slightly different function
//          that avoids the hypotenuse calculation, using NLopt to minimize
//          the sum of the squares of the errors in nu with the constraint
//          that the estimated nu be >= minimum nu to attain machine precision.
//          I also separate the regions where nu == 2 and nu == 1. */
//       const ftype ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
//       ftype xs = y < 0 ? -cre(z) : cre(z); // compute for -z if y < 0
//       if (x + ya > 4000) { // nu <= 2
//           if (x + ya > 1e7) { // nu == 1, w(z) = i/sqrt(pi) / z
//               // scale to avoid overflow
//               if (x > ya) {
//                   faddeeva_algorithm += 1;
//                   ftype yax = ya / xs;
//                   faddeeva_algorithm = 100;
//                   ftype denom = ispi / (xs + yax*ya);
//                   ret = C(denom*yax, denom);
//               }
//               else if (isinf(ya)) {
//                   faddeeva_algorithm += 2;
//                   return ((isnan(x) || y < 0)
//                           ? C(NaN,NaN) : C(0,0));
//               }
//               else {
//                   faddeeva_algorithm += 3;
//                   ftype xya = xs / ya;
//                   ftype denom = ispi / (xya*xs + ya);
//                   ret = C(denom, denom*xya);
//               }
//           }
//           else { // nu == 2, w(z) = i/sqrt(pi) * z / (z*z - 0.5)
//               faddeeva_algorithm += 4;
//               ftype dr = xs*xs - ya*ya - 0.5, di = 2*xs*ya;
//               ftype denom = ispi / (dr*dr + di*di);
//               ret = C(denom * (xs*di-ya*dr), denom * (xs*dr+ya*di));
//           }
//       }
//       else { // compute nu(z) estimate and do general continued fraction
//           faddeeva_algorithm += 5;
//           const ftype c0=3.9, c1=11.398, c2=0.08254, c3=0.1421, c4=0.2023; // fit
//           ftype nu = floor(c0 + c1 / (c2*x + c3*ya + c4));
//           ftype wr = xs, wi = ya;
//           for (nu = 0.5 * (nu - 1); nu > 0.4; nu -= 0.5) {
//               // w <- z - nu/w:
//               ftype denom = nu / (wr*wr + wi*wi);
//               wr = xs - wr * denom;
//               wi = ya + wi * denom;
//           }
//           { // w(z) = i/sqrt(pi) / w:
//               ftype denom = ispi / (wr*wr + wi*wi);
//               ret = C(denom*wi, denom*wr);
//           }
//       }
//       if (y < 0) {
//           faddeeva_algorithm += 10;
//           // use w(z) = 2.0*exp(-z*z) - w(-z),
//           // but be careful of overflow in exp(-z*z)
//           //                                = exp(-(xs*xs-ya*ya) -2*i*xs*ya)
//           return csub(cmul(C(2,0), cexp(C((ya-xs)*(xs+ya), 2*xs*y))), ret);
//       }
//       else
//           return ret;
//   }
// 
//   /* Note: The test that seems to be suggested in the paper is x <
//      sqrt(-log(DBL_MIN)), about 26.6, since otherwise exp(-x^2)
//      underflows to zero and sum1,sum2,sum4 are zero.  However, long
//      before this occurs, the sum1,sum2,sum4 contributions are
//      negligible in ftype precision; I find that this happens for x >
//      about 6, for all y.  On the other hand, I find that the case
//      where we compute all of the sums is faster (at least with the
//      precomputed expa2n2 table) until about x=10.  Furthermore, if we
//      try to compute all of the sums for x > 20, I find that we
//      sometimes run into numerical problems because underflow/overflow
//      problems start to appear in the various coefficients of the sums,
//      below.  Therefore, we use x < 10 here. */
//   else if (x < 10) {
// 
//       faddeeva_algorithm = 200;
// 
//       ftype prod2ax = 1, prodm2ax = 1;
//       ftype expx2;
// 
//       if (isnan(y)) {
//           faddeeva_algorithm += 99;
//           return C(y,y);
//       }
// 
//       if (x < 5e-4) { // compute sum4 and sum5 together as sum5-sum4
//                       // This special case is needed for accuracy.
//           faddeeva_algorithm += 1;
//           const ftype x2 = x*x;
//           expx2 = 1 - x2 * (1 - 0.5*x2); // exp(-x*x) via Taylor
//           // compute exp(2*a*x) and exp(-2*a*x) via Taylor, to ftype precision
//           const ftype ax2 = 1.036642960860171859744*x; // 2*a*x
//           const ftype exp2ax =
//               1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667*ax2));
//           const ftype expm2ax =
//               1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667*ax2));
//           for (int n = 1; ; ++n) {
//               ++faddeeva_nofterms;
//               const ftype coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
//               prod2ax *= exp2ax;
//               prodm2ax *= expm2ax;
//               sum1 += coef;
//               sum2 += coef * prodm2ax;
//               sum3 += coef * prod2ax;
// 
//               // really = sum5 - sum4
//               sum5 += coef * (2*a) * n * __sinh_taylor((2*a)*n*x);
// 
//               // test convergence via sum3
//               if (coef * prod2ax < DBLEPS * sum3) break;
//           }
//       }
//       else { // x > 5e-4, compute sum4 and sum5 separately
//           faddeeva_algorithm += 2;
//           expx2 = exp(-x*x);
//           const ftype exp2ax = exp((2*a)*x), expm2ax = 1 / exp2ax;
//           for (int n = 1; ; ++n) {
//               ++faddeeva_nofterms;
//               const ftype coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
//               prod2ax *= exp2ax;
//               prodm2ax *= expm2ax;
//               sum1 += coef;
//               sum2 += coef * prodm2ax;
//               sum4 += (coef * prodm2ax) * (a*n);
//               sum3 += coef * prod2ax;
//               sum5 += (coef * prod2ax) * (a*n);
//               // test convergence via sum5, since this sum has the slowest decay
//               if ((coef * prod2ax) * (a*n) < DBLEPS * sum5) break;
//           }
//       }
//       const ftype expx2erfcxy = // avoid spurious overflow for large negative y
//           y > -6 // for y < -6, erfcx(y) = 2*exp(y*y) to ftype precision
//           ? expx2*erfcx(y) : 2*exp(y*y-x*x);
//       if (y > 5) { // imaginary terms cancel
//           faddeeva_algorithm += 10;
//           const ftype sinxy = sin(x*y);
//           ret = C( (expx2erfcxy - c*y*sum1) * cos(2*x*y)
//               + (c*x*expx2) * sinxy * __sinc(x*y, sinxy), 0.);
//       }
//       else {
//           faddeeva_algorithm += 20;
//           ftype xs = cre(z);
//           const ftype sinxy = sin(xs*y);
//           const ftype sin2xy = sin(2*xs*y), cos2xy = cos(2*xs*y);
//           const ftype coef1 = expx2erfcxy - c*y*sum1;
//           const ftype coef2 = c*xs*expx2;
//           ret = C(coef1 * cos2xy + coef2 * sinxy * __sinc(xs*y, sinxy),
//                   coef2 * __sinc(2*xs*y, sin2xy) - coef1 * sin2xy);
//       }
//   }
//   else
//   {
//     // x large: only sum3 & sum5 contribute (see above note)
//     faddeeva_algorithm = 300;
// 
//     if (isnan(x)) { return C(x,x); }
//     if (isnan(y)) { return C(y,y); }
// 
//     ret = C(exp(-x*x), 0.0); // |y| < 1e-10, so we only need exp(-x*x) term
//     // (round instead of ceil as in original paper; note that x/a > 1 here)
//     ftype n0 = floor(x/a + 0.5); // sum in both directions, starting at n0
//     ftype dx = a*n0 - x;
//     sum3 = exp(-dx*dx) / (a2*(n0*n0) + y*y);
//     sum5 = a*n0 * sum3;
//     ftype exp1 = exp(4*a*dx), exp1dn = 1;
//     int dn;
//     for (dn = 1; n0 - dn > 0; ++dn) { // loop over n0-dn and n0+dn terms
//         ftype np = n0 + dn, nm = n0 - dn;
//         ftype tp = exp(-sqr(a*dn+dx));
//         ftype tm = tp * (exp1dn *= exp1); // trick to get tm from tp
//         tp /= (a2*(np*np) + y*y);
//         tm /= (a2*(nm*nm) + y*y);
//         sum3 += tp + tm;
//         sum5 += a * (np * tp + nm * tm);
//         if (a * (np * tp + nm * tm) < DBLEPS * sum5) break;//goto finish;
//     }
//     while (1)
//     {
//       // loop over n0+dn terms only (since n0-dn <= 0)
//       ftype np = n0 + dn++;
//       ftype tp = exp(-sqr(a*dn+dx)) / (a2*(np*np) + y*y);
//       sum3 += tp;
//       sum5 += a * np * tp;
//       if (a * np * tp < DBLEPS * sum5) break;//goto finish;
//     }
//   }
// //finish:
//   return ret + C((0.5*c)*y*(sum2+sum3), (0.5*c)*copysign(sum5-sum4, cre(z)));
// }




WITHIN_KERNEL
ctype cwofz( ctype z)
{
   ftype in_real = z.x;
   ftype in_imag = z.y;
   int n, nc, nu;
   ftype h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
   ftype Rx [33];
   ftype Ry [33];

   x = fabs(in_real);
   y = fabs(in_imag);

   if (y < YLIM && x < XLIM) {
      q = (1.0 - y / YLIM) * sqrt(1.0 - (x / XLIM) * (x / XLIM));
      h  = 1.0 / (3.2 * q);
      #ifdef CUDA
        nc = 7 + int(23.0 * q);
      #else
        nc = 7 + convert_int(23.0 * q);
      #endif

//       xl = pow(h, ftype(1 - nc));
      ftype h_inv = 1./h;
      xl = h_inv;
      for(int i = 1; i < nc-1; i++)
          xl *= h_inv;

      xh = y + 0.5 / h;
      yh = x;
      #ifdef CUDA
        nu = 10 + int(21.0 * q);
      #else
        nu = 10 + convert_int(21.0 * q);
      #endif
      Rx[nu] = 0.;
      Ry[nu] = 0.;
      for (n = nu; n > 0; n--){
         Tx = xh + n * Rx[n];
         Ty = yh - n * Ry[n];
         Tn = Tx*Tx + Ty*Ty;
         Rx[n-1] = 0.5 * Tx / Tn;
         Ry[n-1] = 0.5 * Ty / Tn;
         }
      Sx = 0.;
      Sy = 0.;
      for (n = nc; n>0; n--){
         Saux = Sx + xl;
         Sx = Rx[n-1] * Saux - Ry[n-1] * Sy;
         Sy = Rx[n-1] * Sy + Ry[n-1] * Saux;
         xl = h * xl;
      };
      Wx = ERRF_CONST * Sx;
      Wy = ERRF_CONST * Sy;
   }
   else {
      xh = y;
      yh = x;
      Rx[0] = 0.;
      Ry[0] = 0.;
      for (n = 9; n>0; n--){
         Tx = xh + n * Rx[0];
         Ty = yh - n * Ry[0];
         Tn = Tx * Tx + Ty * Ty;
         Rx[0] = 0.5 * Tx / Tn;
         Ry[0] = 0.5 * Ty / Tn;
      };
      Wx = ERRF_CONST * Rx[0];
      Wy = ERRF_CONST * Ry[0];
   }

   if (y == 0.) {
      Wx = exp(-x * x);
   }
   if (in_imag < 0.) {

      ftype exp_x2_y2 = exp(y * y - x * x);
      Wx =   2.0 * exp_x2_y2 * cos(2.0 * x * y) - Wx;
      Wy = - 2.0 * exp_x2_y2 * sin(2.0 * x * y) - Wy;
      if (in_real > 0.) {
         Wy = -Wy;
      }
   }
   else if (in_real < 0.) {
      Wy = -Wy;
   }

   return C(Wx,Wy);
}


/*
WITHIN_KERNEL
ctype ipanema_erfc2(ctype z)
{
  ftype re = -z.x * z.x + z.y * z.y;
  ftype im = -2. * z.x * z.y;
  ctype expmz = cexp( C(re,im) );

  if (z.x >= 0.0) {
    return                 cmul( expmz, faddeeva(C(-z.y,+z.x)) );
  }
  else{
    ctype ans = cmul( expmz, faddeeva(C(+z.y,-z.x)) );
    return C(2.0-ans.x, ans.y);
  }
}



WITHIN_KERNEL
ctype ipanema_erfc(ctype z)
{
  if (z.y<0)
  {
    ctype ans = ipanema_erfc2( C(-z.x, -z.y) );
    return C( 2.0-ans.x, -ans.y);
  }
  else{
    return ipanema_erfc2(z);
  }
}
*/


WITHIN_KERNEL
ctype cerfc(ctype x)
{
  ctype z = cmul(I,x);
  ctype result = cmul( cexp(  cmul(C(-1,0),cmul(x,x))   ) , cwofz(z) );

  //printf("z = %+.16f %+.16fi\n", z.x, z.y);
  //printf("fad = %+.16f %+.16fi\n", faddeeva(z).x, faddeeva(z).y);

  if (x.x > 20.0){// && fabs(x.y < 20.0)
    result = C(0.0,0);
  }
  if (x.x < -20.0){// && fabs(x.y < 20.0)
    result = C(2.0,0);
  }

  return result;
}


/*
WITHIN_KERNEL
ctype cerfc(ctype z)
{
  if (z.y<0)
  {
    ctype ans = cErrF_2( C(-z.x, -z.y) );
    return C( 2.0-ans.x, -ans.y);
  }
  else{
    return cErrF_2(z);
  }
}
*/



WITHIN_KERNEL
ctype cerfcx(const ctype z)
{
  return cwofz(C(-cim(z), cre(z)));
}



WITHIN_KERNEL
ftype erfi(const ftype x)
{
  return x*x > 720 ? (x > 0 ? Inf : -Inf) : exp(x*x) * im_w_of_x(x);
}



WITHIN_KERNEL
ftype dawson(const ftype x)
{
  return M_SQRTPIHALF * im_w_of_x(x);
}



WITHIN_KERNEL
ftype rwofzr(const ftype x, const ftype y)
{
  return cre(cwofz(C(x,y)));
}



WITHIN_KERNEL
ftype rwofzi(const ftype x, const ftype y)
{
  return cim(cwofz(C(x,y)));
}



WITHIN_KERNEL
ftype voigt(const ftype x, const ftype sigma, const ftype gamma)
{
  const ftype gam = gamma < 0 ? -gamma : gamma;
  const ftype sig = sigma < 0 ? -sigma : sigma;

  if ( gam==0 )
  {
    if ( sig==0 )
    {
      return (x!=0) ? 0.0 : Inf; // delta
    }
    else
    {
      return exp( -x*x/2/(sig*sig) ) / M_SQRT2PI / sig; // gaussian
    }
  }
  else
  {
    if ( sig==0 )
    {
      return gam / M_PI / (x*x + gam*gam);
    }
    else
    {
      ctype z = cdiv(C(x,gam), C(sqrt(2.)/sig,0));
      return cre( cwofz(z) ) / M_SQRT2PI / sig;
    }
  }
}


/*
WITHIN_KERNEL
ctype cerfc(const ctype z)
{
  const ftype x = cre(z);
  const ftype y = cim(z);

  if (x == 0.)
      return C(1, y*y > 720 ? (y > 0 ? -Inf : Inf) : -exp(y*y) * im_w_of_x(y));
  if (y == 0.)
  {
    if (x*x > 750) // underflow
      return C(x >= 0 ? 0.0 : 2.0, -y);
      return C(x >= 0 ? exp(-x*x) * erfcx(x) : 2. - exp(-x*x) * erfcx(-x), -y);
  }

  const ftype zR = (y - x) * (x + y);
  const ftype zI = -2*x*y;

  if (zR < -750)
  {
    return (x >= 0) ? C(0.0,0.0) : C(2.0,0.0);
  }

  if (x >= 0)
  {
    return cmul( cexp(C(zR, zI)), cwofz(C(-y,x)) );
  }
  else
  {
    return csub(C(2,0), cmul(cexp(C(zR, zI)), cwofz(C(y,-x))) );
  }

}
*/


WITHIN_KERNEL
ctype cerf(const ctype z)
{
  return csub( C(1,0), cerfc(z) );
}



WITHIN_KERNEL
ctype cerfi(const ctype z)
{
  const ctype e = cerf(C(-cim(z),cre(z)));
  return C(cim(e), -cre(e));
}
