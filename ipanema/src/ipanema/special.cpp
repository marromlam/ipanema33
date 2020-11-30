#include "special.hpp"
#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29


WITHIN_KERNEL
ftype factorial(int n)
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
ftype legendre_poly(int l, int m, ftype cos_theta)
{
    if(l == 0 && m == 0)
    {
        return 1.;
    }
    else if(l == 1 && m == 0)
    {
        return cos_theta;
    }
    else if(l == 2 && m == 0)
    {
        return 0.5*(3.*cos_theta*cos_theta - 1.);
    }
    else if(l == 2 && (m == 1 || m == -1))
    {
        return -3.*cos_theta*sqrt(1.-cos_theta*cos_theta);
    }
    else if(l == 2 && (m == 2 || m == -2))
    {
        return 3.*cos_theta*(1.-cos_theta*cos_theta);
    }
    else
        printf("ATTENTION: Legendre polynomial index l,m is out of the range of this function. Check code.");

    return 0.;
}



WITHIN_KERNEL
ftype sph_harm(int l, int m, ftype cos_theta, ftype phi)
{
    if(m == 0)
    {
        return sqrt((2*l + 1)/(4.*M_PI))*legendre_poly(l, m, cos_theta);
    }
    else if(m > 0)
    {
        return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*legendre_poly(l, m, cos_theta)*cos(m*phi);
    }
    else
    {
        return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-(-1.*m)))/sqrt(factorial(l-1.*m)))*legendre_poly(l, -1.*m, cos_theta)*sin(-1.*m*phi);
    }

    return 0.;
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




//Legendre polynomianls up to l = 2
WITHIN_KERNEL
double P_lm(int l, int m, double cos_psi)
{
//     double factor = 1./(l+0.5);
    double factor = 1.;

    if(l == 0 && m == 0)
    {
        return factor*1.;
    }
    else if(l == 1 && m == 0)
    {
        return factor*cos_psi;
    }
    else if(l == 2 && m == 0)
    {
        return factor*0.5*(3.*cos_psi*cos_psi - 1.);
    }
/*
    else if(l == 2 && (m == 1 || m == -1))
    {
        return -factor*3.*cos_psi*sqrt(1.-cos_psi*cos_psi);
    }
*/
    else if(l == 2 && m == 1 )
    {
        return -factor*3*cos_psi*sqrt(1.-cos_psi*cos_psi);
    }
    else if(l == 2 && m == -1 )
    {
        return factor*0.5*cos_psi*sqrt(1.-cos_psi*cos_psi);
    }
/*
    else if(l == 2 && (m == 2 || m == -2))
    {
        return factor*3.*cos_psi*(1.-cos_psi*cos_psi);
    }
*/
    else if(l == 2 && m == 2)
    {
        return factor*3.*(1.-cos_psi*cos_psi);
    }
    else if(l == 2 && m == -2)
    {
        return factor*0.125*(1.-cos_psi*cos_psi);
    }
    else
        printf("ATTENTION: Legendre polynomial index l,m is out of the range of this function. Check code.");

    return 0.;
}

//Spherical harmonics up to l = 2
WITHIN_KERNEL
double Y_lm(int l, int m, double cos_theta, double phi)
{
    double P_l;
//     double factor = 1./(l+0.5);
    double factor = 1.;

    if(l == 0)
    {
        P_l = factor*1.;
    }
    else if (l == 1)
    {
        P_l = factor*cos_theta;
    }
    else if (l == 2)
    {
        P_l = factor*0.5*(3*cos_theta*cos_theta-1.);
    }
    else if (l > 2)
    {
        printf("ATTENTION: Ylm polynomial index l is out of the range of this function. Check code.");
        return 0.;
    }

    if(m == 0)
    {
//         return sqrt((2*l + 1)/(4.*M_PI))*P_lm(l, m, cos_theta);
        return sqrt((2*l + 1)/(4.*M_PI))*P_l;
    }
    else if(m > 0)
    {
//         return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*cos(m*phi);
        return sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*cos(m*phi);
    }
    else
    {
//         return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-(-1.*m)))/sqrt(factorial(l-1.*m)))*P_lm(l, -1.*m, cos_theta)*sin(-1.*m*phi);
      m = abs(m);
        return sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*sin(m*phi);
    }

    return 0.;
}




















KERNEL
void pywofz(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = faddeeva(z[idx]);
}



KERNEL
void pyfaddeeva(GLOBAL_MEM ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = faddeeva(z[idx]);
}


KERNEL
void pycerfc(GLOBAL_MEM ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = cerfc(z[idx]);
   printf("erfc(%+.4f%+.4fi) = %+.4f%+.4fi\n",z[idx].x,z[idx].y,out[idx].x,out[idx].y);
}

KERNEL
void pyipacerfc(GLOBAL_MEM ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = ipanema_erfc(z[idx]);
}
