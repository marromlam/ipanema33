#include "special.hpp"
#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29


WITHIN_KERNEL
ftype factorial(const int n)
{
   if(n <= 0)
    return 1.;

   ftype x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}



WITHIN_KERNEL
ctype faddeeva( ctype z)
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

   return cnew(Wx,Wy);
}



WITHIN_KERNEL
ftype lpmv(const int l, const int m, const ftype cosT)
{
    const int L = (l<0) ? abs(l)-1 : l; 
    const int M = abs(m);
    ftype factor = 1.0;

    if (m<0){
        factor = pow(-1.0,(ftype) m) * factorial(L-M) / factorial(L+M);
    }

    // shit
    if (M>l){
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
        if (get_global_id(0) < 10) {
          printf("WARNING: Associated Legendre polynomial (%+d,%+d) is out of the scope of this function.", l, m);
        }
        return 0;
    }

}



WITHIN_KERNEL
ctype csph_harm(const int l, const int m, const ftype cosT, const ftype phi)
{
  ftype ans = lpmv(l, m, cosT);
  ans *= sqrt( ((2*l+1)*factorial(l-m)) / (4*M_PI*factorial(l+m)) );
  return cnew(ans*cos(m*phi), ans*sin(m*phi));
}





WITHIN_KERNEL
ftype sph_harm(const int l, const int m, const ftype cosT, const ftype phi)
{
    if(m < 0)
    {
      return pow(-1.,m) * sqrt(2.) * cim( csph_harm(l, -m, cosT, phi) );
    }
    else if(m > 0)
    {
      return pow(-1.,m) * sqrt(2.) * cre( csph_harm(l,  m, cosT, phi) );
    }
    else
    {
      return sqrt( (2.*l+1.) / (4.*M_PI) ) * lpmv(l, m, cosT);
    }
}



WITHIN_KERNEL
ctype ipanema_erfc2(ctype z)
{
  ftype re = -z.x * z.x + z.y * z.y;
  ftype im = -2. * z.x * z.y;
  ctype expmz = cexp( cnew(re,im) );

  if (z.x >= 0.0) {
    return                 cmul( expmz, faddeeva(cnew(-z.y,+z.x)) );
  }
  else{
    ctype ans = cmul( expmz, faddeeva(cnew(+z.y,-z.x)) );
    return cnew(2.0-ans.x, ans.y);
  }
}



WITHIN_KERNEL
ctype ipanema_erfc(ctype z)
{
  if (z.y<0)
  {
    ctype ans = ipanema_erfc2( cnew(-z.x, -z.y) );
    return cnew( 2.0-ans.x, -ans.y);
  }
  else{
    return ipanema_erfc2(z);
  }
}



WITHIN_KERNEL
ctype cErrF_2(ctype x)
{
  ctype I = cnew(0.0,1.0);
  ctype z = cmul(I,x);
  ctype result = cmul( cexp(  cmul(cnew(-1,0),cmul(x,x))   ) , faddeeva(z) );

  //printf("z = %+.16f %+.16fi\n", z.x, z.y);
  //printf("fad = %+.16f %+.16fi\n", faddeeva(z).x, faddeeva(z).y);

  if (x.x > 20.0){// && fabs(x.y < 20.0)
    result = cnew(0.0,0);
  }
  if (x.x < -20.0){// && fabs(x.y < 20.0)
    result = cnew(2.0,0);
  }

  return result;
}


WITHIN_KERNEL
ctype cerfc(ctype z)
{
  if (z.y<0)
  {
    ctype ans = cErrF_2( cnew(-z.x, -z.y) );
    return cnew( 2.0-ans.x, -ans.y);
  }
  else{
    return cErrF_2(z);
  }
}


WITHIN_KERNEL
ftype tarasca(ftype n) {
  // Integrate[x^n*Sqrt[1 - x^2], {x, -1, 1}]
  ftype ans = (1 + pow(-1.,n)) * sqrt(M_PI) * tgamma((1 + n)/2);
  return ans / (4.*tgamma(2 + n/2));
}



WITHIN_KERNEL
ftype curruncho(ftype n, ftype m, ftype xi, ftype xf) {
  // Integrate[x^n*Cos[x]^m, {x, -1, 1}]
  ftype ans = 0;
  ftype kf = 0;
  ftype mupp = floor((m+1)/2);
  ftype mlow = floor((m-1)/2);
  for (int k=0; k<mlow; k++){
    kf = k; 
    ans += pow(-1., kf) * (tgamma(m+1)/tgamma(m-2*k)) * pow(n*M_PI, m-2*kf-1);
  }
  ans *= pow(-1.0, n) / pow(n, m+1);
  ans += pow(-1,mupp) * (tgamma(m+1)*floor(2*mupp - m))/pow(n, m+1);
  return ans;
}



WITHIN_KERNEL
ftype pozo(ftype n, ftype m, ftype xi, ftype xf)  {
  // Integrate[x^n*Sin[x]^m, {x, -1, 1}]
  ftype ans = 0;
  ftype kf = 0;
  ftype mhalf = floor(m/2);
  for (int k=0; k<mhalf; k++){
     kf = k;
     ans += pow(-1., kf) * (tgamma(m+1)/tgamma(m-2*k+1)) * pow(n*M_PI, m-2*k);
  }
  ans *= pow(-1., n+1) / pow(n, m+1);
  ans -= pow(-1, mhalf) * (tgamma(m+1)*floor(m-2*mhalf-1)) / pow(n, m+1);
  return ans;
}







WITHIN_KERNEL
ftype __core_igamc(ftype a, ftype x)
{
    const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
    const ftype BIG = 4.503599627370496e15;
    const ftype BIGINV = 2.22044604925031308085e-16;

    ftype ans, ax, c, yc, r, t, y, z;
    ftype pk, pkm1, pkm2, qk, qkm1, qkm2;

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ax = a * log(x) - x - lgamma(a);
    if (ax < -MAXLOG) return 0.0;  // underflow
    ax = exp(ax);

    /* continued fraction */
    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1/qkm1;

    do {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        yc = y * c;
        pk = pkm1 * z  -  pkm2 * yc;
        qk = qkm1 * z  -  qkm2 * yc;
        if (qk != 0) {
            r = pk/qk;
            t = fabs( (ans - r)/r );
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (fabs(pk) > BIG) {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
    } while( t > MACHEP );

    return( ans * ax );
}




WITHIN_KERNEL
ftype __core_igam(ftype a, ftype x)
{
    const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
    ftype ans, ax, c, r;

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ax = a * log(x) - x - lgamma(a);
    if (ax < -MAXLOG) return 0.0; // underflow
    ax = exp(ax);

    /* power series */
    r = a;
    c = 1.0;
    ans = 1.0;

    do {
        r += 1.0;
        c *= x/r;
        ans += c;
    } while (c/ans > MACHEP);

    return ans * ax/a;
}








WITHIN_KERNEL
ftype gammaln(ftype x);

WITHIN_KERNEL
ftype gammainc(ftype a, ftype x);

WITHIN_KERNEL
ftype gammaincc(ftype a, ftype x);




WITHIN_KERNEL
ftype gammaln(ftype x)
{
  if (isnan(x)) return(x);
  if (!isfinite(x)) return(INFINITY);
  return lgamma(x);
}

WITHIN_KERNEL
ftype gammainc(ftype a, ftype x)
{
    if ((x <= 0) || ( a <= 0)) return 0.0;
    if ((x > 1.0) && (x > a)) return 1.0 - __core_igamc(a, x);
    return __core_igam(a, x);
}

WITHIN_KERNEL
ftype gammaincc(ftype a, ftype x)
{
    if ((x <= 0.0) || (a <= 0)) return 1.0;
    if ((x <  1.0) || (x <  a)) return 1.0 - __core_igam(a, x);
    return __core_igamc(a, x);
}





























































KERNEL
void pywofz(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = faddeeva(z[idx]);
}



KERNEL
void pyfaddeeva(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = faddeeva(z[idx]);
}


KERNEL
void pycerfc(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = cerfc(z[idx]);
}

KERNEL
void pyipacerfc(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = ipanema_erfc(z[idx]);
}

KERNEL
void pylpmv(const int l, const int m, GLOBAL_MEM const ftype *cos_theta, 
            GLOBAL_MEM ftype *out)
{
  int idx = get_global_id(0);
  out[idx] = lpmv(l,m,cos_theta[idx]);
}

KERNEL
void pytessel_sph_harm(const int l, const int m, 
                       GLOBAL_MEM const ftype *cos_theta,
                       GLOBAL_MEM const ftype *phi, GLOBAL_MEM ftype *out)
{
  int idx = get_global_id(0);
  out[idx] = sph_harm(l, m, cos_theta[idx], phi[idx]);
}

KERNEL
void pysph_harm(const int l, const int m, GLOBAL_MEM const ftype *cosT, 
                GLOBAL_MEM const ftype *phi, GLOBAL_MEM ctype *out)
{
  int idx = get_global_id(0);
  out[idx] = csph_harm(l, m, cosT[idx], phi[idx]);
}
