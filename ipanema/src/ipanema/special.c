#include "core.h"
#include "special.h"


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// FACTORIAL                                                                  //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
int nfactorial(const int n)
{
   if (n <= 0) { return 1.; }

   int x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}



WITHIN_KERNEL
ftype factorial(const int n)
{
   if (n <= 0) { return 1.; }

   ftype x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}


/**
 * Returns ( n / k ) binomial coefficient.
 */
WITHIN_KERNEL
ftype binom(const int n, const int k)
{
  return factorial(n)/(factorial(k)*factorial(n-k));
}




////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE ERROR FUNCTION FAMILY                                                  //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ftype erfcx(const ftype x)
{
    // Steven G. Johnson, October 2012.

    // This function combines a few different ideas.

    // First, for x > 50, it uses a continued-fraction expansion (same as
    // for the Faddeeva function, but with algebraic simplifications for z=i*x).

    // Second, for 0 <= x <= 50, it uses Chebyshev polynomial approximations,
    // but with two twists:
    //
    // a) It maps x to y = 4 / (4+x) in [0,1].  This simple transformation,
    // inspired by a similar transformation in the octave-forge/specfun
    // erfcx by Soren Hauberg, results in much faster Chebyshev convergence
    // than other simple transformations I have examined.
    //
    // b) Instead of using a single Chebyshev polynomial for the entire
    // [0,1] y interval, we break the interval up into 100 equal
    // subintervals, with a switch/lookup table, and use much lower
    // degree Chebyshev polynomials in each subinterval. This greatly
    // improves performance in my tests.
    //
    // For x < 0, we use the relationship erfcx(-x) = 2 exp(x^2) - erfc(x),
    // with the usual checks for overflow etcetera.

    // Performance-wise, it seems to be substantially faster than either
    // the SLATEC DERFC function [or an erfcx function derived therefrom]
    // or Cody's CALERF function (from netlib.org/specfun), while
    // retaining near machine precision in accuracy.

    if (x >= 0) {
        if (x > 50) { // continued-fraction expansion is faster
            const ftype ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
            if (x > 5e7) // 1-term expansion, important to avoid overflow
                return ispi / x;
            /* 5-term expansion (rely on compiler for CSE), simplified from:
               ispi / (x+0.5/(x+1/(x+1.5/(x+2/x))))  */
            return ispi*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75));
        }
        return erfcx_y100(400/(4+x));
    }
    else
        return x < -26.7 ? HUGE_VAL : (x < -6.1 ? 2*exp(x*x)
                                       : 2*exp(x*x) - erfcx_y100(400/(4-x)));
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  POLYNOMIALS and friends                                                   //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ftype lpmv(const int m, const int l, const ftype cosT)
{
    const int L = (l<0) ? abs(l)-1 : l;
    const int M = abs(m);
    ftype factor = 1.0;

    if (m<0){
        factor = pow(-1.0, 1.*m) * factorial(L-M) / factorial(L+M);
    }

    // shit
    if (M>l){
        /* printf("WARNING: Associated Legendre polynomial (%+d,%+d) is out of the scope of this function.", l, m); */
        return 0;
    }

    // L = 0
    if (L==0)
    {
        return 1.0;
    }
    // L = 1
    else if (L==1)
    {
        if      (M==0) { return cosT; }
        else           { return -factor*sqrt(1.0-cosT*cosT); } // OK
    }
    // L = 2
    else if (L==2)
    {
        if      (M==0) { return  0.5*(3.*cosT*cosT - 1.); } // OK
        else if (M==1) { return -3.0*factor*cosT*sqrt(1.-cosT*cosT); } // OK
        else           { return  3.0*factor*(1.-cosT*cosT); } // OK
    }
    // L = 3
    else if (L==3)
    {
        ftype sinT = sqrt(1.0-cosT*cosT);
        if      (M==0) { return   0.5*(5.*cosT*cosT*cosT - 3.*cosT); }
        else if (M==1) { return  -1.5*factor*(5.*cosT*cosT - 1.)*sinT; }
        else if (M==2) { return  15.0*factor*sinT*sinT*cosT; }
        else           { return -15.0*factor*sinT*sinT*sinT; }
    }
    // L = 4
    else if (L==4)
    {
        ftype sinT = sqrt(1.0-cosT*cosT);
        if      (M==0) { return 0.125*(35.*cosT*cosT*cosT*cosT - 30.*cosT*cosT + 3.); }
        else if (M==1) { return  -2.5*factor*(7.*cosT*cosT*cosT - 3.*cosT)*sinT; }
        else if (M==2) { return   7.5*factor*(7.*cosT*cosT - 1.)*sinT*sinT; }
        else if (M==3) { return -105.*factor*sinT*sinT*sinT*cosT; }
        else           { return  105.*factor*sinT*sinT*sinT*sinT; }
    }
    else {
        /* if (get_global_id(0) < 10000000000) { */
        /*   printf("WARNING: Associated Legendre polynomial (%+d,%+d) is out of the scope of this function.", l, m); */
        /* } */
        //asm(“trap;”);
        return 0;
    }

}


/*  NEEED TO FIX THIS
WITHIN_KERNEL
ftype lpmv(const int M, const int l, const ftype x)
{
  //const int l = (L<0) ? abs(L)-1 : L;
  const int m = abs(M);

  ftype factor = 1.0;

  ftype fact, pll, pmm, pmmp1, somx2; 
  int i, ll;

  if (m < 0 || m > l || fabs(x) > 1.0) 
    printf("Wrong arguments in routine lpmv\n");
 
  pmm = 1;
  if (m > 0)
  {
    somx2 = sqrt((1.0-x)*(1.0+x));
    fact = 1.0;
    for (i=1; i<=m; i++)
    {
      pmm *= -fact*somx2;
      fact += 2.0;
    }
  }
  
  if (l == m)
  {
    return pmm;
  }
  else
  {
    pmmp1 = x* (2*m+1) * pmm;
    if (l == (m+1))
    {
      return pmmp1;
    }
    else
    {
      for (ll=m+2; ll<=l; ll++)
      {
        pll = (x*(2*ll-1)*pmmp1-(ll+m-1)*pmm) / (ll-m);
        pmm = pmmp1;
        pmmp1 = pll;
      }
      return pll;
    }
  }
}
*/



WITHIN_KERNEL
ftype sph_harm(const int m, const int l, const ftype cosT, const ftype phi)
{
    if(m < 0)
    {
      return pow(-1.,m) * sqrt(2.) * cim( csph_harm(-m, l, cosT, phi) );
    }
    else if(m > 0)
    {
      return pow(-1.,m) * sqrt(2.) * cre( csph_harm(m,  l, cosT, phi) );
    }
    else
    {
      return sqrt( (2.*l+1.) / (4.*M_PI) ) * lpmv(m, l, cosT);
    }
}



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  THE GAMMA FAMILY                                                          //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ftype rgamma(ftype x)
{
  #ifdef CUDA
    return tgammaf(x);
  #else
    return tgamma(x);
  #endif
}


/*
WITHIN_KERNEL
ftype tarasca(ftype n)
{
  // Integrate[x^n*Sqrt[1 - x^2], {x, -1, 1}]
  ftype ans = (1 + pow(-1.,n)) * sqrt(M_PI) * rgamma((1 + n)/2);
  return ans / (4.*rgamma(2 + n/2));
}



WITHIN_KERNEL
ftype curruncho(const ftype n, const ftype m, const ftype xi, const ftype xf) {
  // Integrate[x^n*Cos[x]^m, {x, -1, 1}]
  ftype ans = 0;
  if (xi == xf) return ans;
  if (n == 0.0) return pow(xf, m+1.)/(m+1.) - pow(xi, m+1.)/(m+1.);

  ftype kf = 0;
  ftype mupp = floor((m+1)/2);
  ftype mlow = floor((m-1)/2);
  for (int k=0; k<mlow; k++){
    kf = k;
    ans += pow(-1., kf) * (rgamma(m+1)/rgamma(m-2*k)) * pow(n*M_PI, m-2*kf-1);
  }
  ans *= pow(-1., n) / pow(n, m+1);
  ans += pow(-1.,mupp) * (rgamma(m+1)*floor(2*mupp - m))/pow(n, m+1);
  return ans*(xf - pow(-1.,m)*xi)/M_PI;
}



WITHIN_KERNEL
ftype pozo(const ftype n, const ftype m, const ftype xi, const ftype xf)  {
  // Integrate[x^n*Sin[x]^m, {x, -1, 1}]
  ftype ans = 0;
  if (xi == xf) return ans;
  if (n == 0.0) return ans;

  ftype kf = 0;
  ftype mhalf = floor(m/2);
  for (int k=0; k<mhalf; k++){
     kf = k;
     ans += pow(-1., kf) * (rgamma(m+1)/rgamma(m-2*k+1)) * pow(n*M_PI, m-2*k);
  }
  ans *= pow(-1., n+1) / pow(n, m+1);
  ans -= pow(-1., mhalf) * (rgamma(m+1)*floor(m-2*mhalf-1)) / pow(n, m+1);
  return ans*(xf - pow(-1.,m)*xi)/M_PI;
}
*/














WITHIN_KERNEL
ftype rgammaln(const ftype x)
{
  if (isnan(x)) return(x);
  if (!isfinite(x)) return(INFINITY);
  return lgamma(x);
}



WITHIN_KERNEL
ftype rgammainc(const ftype a, const ftype x)
{
    if ((x <= 0) || ( a <= 0)) return 0.0;
    if ((x > 1.0) && (x > a)) return 1.0 - __core_igamc(a, x);
    return __core_igam(a, x);
}



WITHIN_KERNEL
ftype rgammaincc(const ftype a, const ftype x)
{
    if ((x <= 0.0) || (a <= 0)) return 1.0;
    if ((x <  1.0) || (x <  a)) return 1.0 - __core_igam(a, x);
    return __core_igamc(a, x);
}



/*
WITHIN_KERNEL
ftype corzo(ftype n, ftype xi, ftype xf)
{
  // Integrate[x^n*Cos[x]*Cos[x], {x, xi, xf}]
  ctype cte = C(0., +pow(2.,-n) );

  ftype fi = (4*xi)/(1+n);
  fi -= rgamma(1+n)*cre(
                        cmul(
                            cmul(cte, cpow(C(0.,-xi), C(-n,0.))),
                            cgammaincc(1+n, C(0.,-2*xi))
                            )
                        );
  fi += rgamma(1+n)*cre(
                        cmul(
                            cmul(cte, cpow(C(0.,+xi), C(-n,0.))),
                            cgammaincc(1+n, C(0.,+2*xi))
                            )
                        );

  ftype ff = (4*xf)/(1+n);
  ff -= rgamma(1+n)*cre(
                        cmul(
                            cmul(cte, cpow(C(0.,-xf), C(-n,0.))),
                            cgammaincc(1+n, C(0.,-2*xf))  )  );
  ff += rgamma(1+n)*cre(
                        cmul(
                            cmul(cte, cpow(C(0.,+xf), C(-n,0.))),
                            cgammaincc(1+n, C(0.,+2*xf))
                            )
                        );

  return 0.125*( pow(xf,n)*ff - pow(xi,n)*fi );
}



WITHIN_KERNEL
ftype maycar(ftype n, ftype xi, ftype xf)
{
  // Integrate[x^n*Sin[x]*Sin[x], {x, xi, xf}]
  ctype cte = C(0., +pow(2.,-n) );

  ftype fi = pow(2.,2.+n)*xi*pow(xi*xi,n);
  fi += rgamma(1+n)*cre(
                        cmul(
                            cmul(C(0.,1+n), cpow(C(0.,+xi), C(n,0.))),
                            cgammaincc(1+n, C(0.,-2*xi))
                            )
                        );
  fi -= rgamma(1+n)*cre(
                        cmul(
                            cmul(C(0.,1+n), cpow(C(0.,-xi), C(n,0.))),
                            cgammaincc(1+n, C(0.,+2*xi))
                            )
                        );

  ftype ff = pow(2.,2.+n)*xf*pow(xf*xf,n);
  ff += rgamma(1+n)*cre(
                        cmul(
                            cmul(C(0.,1+n), cpow(C(0.,+xf), C(n,0.))),
                            cgammaincc(1+n, C(0.,-2*xf))  )  );
  ff -= rgamma(1+n)*cre(
                        cmul(
                            cmul(C(0.,1+n), cpow(C(0.,-xf), C(n,0.))),
                            cgammaincc(1+n, C(0.,+2*xf))
                            )
                        );

  return (pow(2.,-3-n)/(1+n))*( pow(xf,-n)*ff - pow(xi,-n)*fi );
}
*/



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  THE BESSEL FAMILY                                                         //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ftype rjv(const ftype n, const ftype x)
{
  #ifdef CUDA
    return jnf(n, x);
  #else
    return 0.0;
  #endif
}



WITHIN_KERNEL
ftype riv(const ftype n, const ftype x)
{
  ftype ri, rk, rip, rkp;
  //                      * this is where BesselK is stored!
  bessel_rikv(x, fabs(n), &ri, &rk, &rip, &rkp);
  return ri;
}



WITHIN_KERNEL
ftype rkn(const int m, const ftype x)
{
  // The range is partitioned into the two intervals [0,9.55] and
  // (9.55, infinity). An ascending power series is used in the
  // low range, and an asymptotic expansion in the high range.
  // This function has its biggest error 1e-10 at 9.55.

  ftype k, kf, nk1f, nkf, zn, t, s, z0, z;
  ftype fn, pn, pk, zmn, tlg, tox;
  int i;

  // Index needs to be positive
  int n = (m<0) ? -m : m;

  // Filter some posible problems
  if ( n > MAXFAC ) { 
      /* printf("OVERFLOW\n"); */
      return MAXNUM;
  }
  if ( x <= 0.0 ) {
      /* printf("fora do dominio\n"); */
      return MAXNUM;
  }


  if( x <= 9.55 )
  {
    // An ascending power series
    ftype ans = 0.0;
    z0 = 0.25 * x * x;
    fn = 1.0;
    pn = 0.0;
    zmn = 1.0;
    tox = 2.0/x;

    if( n > 0 )
    {
      // compute factorial of n and psi(n)
      pn = -EUL;
      k = 1.0;
      for( i=1; i<n; i++ )
      {
        pn += 1.0/k;
        k += 1.0;
        fn *= k;
      }

      zmn = tox;

      if( n == 1 )
      {
        ans = 1.0/x;
      }
      else
      {
        nk1f = fn/n;
        kf = 1.0;
        s = nk1f;
        z = -z0;
        zn = 1.0;
        for( i=1; i<n; i++ )
        {
          nk1f = nk1f/(n-i);
          kf = kf * i;
          zn *= z;
          t = nk1f * zn / kf;
          s += t;
          if ( (MAXNUM - fabs(t)) < fabs(s) )
          {
            /* printf("OVERFLOW\n"); */
            return MAXNUM;
          }
          if ( (tox > 1.0) && ((MAXNUM/tox) < zmn) )
          {
            /* printf("OVERFLOW\n"); */
            return MAXNUM;
          }
          zmn *= tox;
        }
        s *= 0.5;
        t = fabs(s);
        if ( (zmn > 1.0) && ((MAXNUM/zmn) < t) )
        {
          //printf("OVERFLOW\n");
          return MAXNUM;
        }
        if ( (t > 1.0) && ((MAXNUM/t) < zmn) )
        {
          //printf("OVERFLOW\n");
          return MAXNUM;
        }
        ans = s * zmn;
      }
    }

    tlg = 2.0 * log( 0.5 * x );
    pk = -EUL;
    if( n == 0 )
    {
      pn = pk;
      t = 1.0;
    }
    else
    {
      pn = pn + 1.0/n;
      t = 1.0/fn;
    }
    s = (pk+pn-tlg)*t;
    k = 1.0;
    do {
      t *= z0 / (k * (k+n));
      pk += 1.0/k;
      pn += 1.0/(k+n);
      s += (pk+pn-tlg)*t;
      k += 1.0;
    } while( fabs(t/s) > MACHEP );

    s = ( n & 1 ) ? -0.5*s/zmn : 0.5*s/zmn;
    return ans + s ;
  }
  else
  {
    // Asymptotic expansion for rkv(x) which converges to 1.4e-17 for x > 18.4

    // filter the underflow
    if ( x > MAXLOG ) { 
        //printf("UNDERFLOW\n");
        return 0.0; 
    }

    k = n;
    pn = 4.0 * k * k;
    pk = 1.0;
    z0 = 8.0 * x;
    fn = 1.0;
    t = 1.0;
    s = t;
    nkf = MAXNUM;
    i = 0;
    do
    {
      z = pn - pk * pk;
      t = t * z /(fn * z0);
      nk1f = fabs(t);
      if ( (i >= n) && (nk1f > nkf) )
      {
        return exp(-x) * sqrt( M_PI/(2.0*x) ) * s;
      }
      nkf = nk1f;
      s += t;
      fn += 1.0;
      pk += 2.0;
      i += 1;
    } while( fabs(t/s) > MACHEP );

    return exp(-x) * sqrt( M_PI/(2.0*x) ) * s;
  }
}



WITHIN_KERNEL
ftype rkv(const ftype n, const ftype x)
{
  // check domain
  if ( x <= 0.0 )
  {
    /* if (x < 0.0) { printf("fora do dominio\n"); } */
    return MAXNUM;
  }

  const ftype nhalf = fract(n);
  if (nhalf == 0.5)
  {
    // Expansion applies for semi-integer n = m - 1/2. We basically compute
    // rkv(0.5,x), and then build the expansion
    //     Sum[((m + k)!/(2^k*k!*(m - k)!))/x^k, {k, m, 0, -1}]
    const int m = (floor(n)<0) ? -floor(n)-1 : floor(n);
    //printf("n=%f --> m, nhalf = %d, %f\n", n, m, nhalf);
    ftype ans = 0.0;
    const ftype kn_nhalf_x = sqrt(M_PI/2.0) * exp(-x) / sqrt(x);
    for (int k=0; k<=m; k++)
    {
      ans += (factorial(m+k)/(pow(2.0,k)*factorial(k)*factorial(m-k)))/pow(x,k);
    }
    return ans * kn_nhalf_x;
  }
  else if (nhalf == 0)
  {
    const int m = floor(n);
    // approximation
    if ( (x < 1.e-06 && m > 0) ||
         (x < 1.e-04 && m > 0 && m < 55) ||
         (x < 0.1    && m >= 55) )
    {
      return rgamma(m) * pow(2.0, m-1) * pow(x, -m);
    }
    return rkn(m, x);
  }
  else
  {
    ftype ri, rk, rip, rkp;
    //                           * this is where BesselK is stored!
    bessel_rikv(x, fabs(n), &ri, &rk, &rip, &rkp);
    return rk;
  }

}
