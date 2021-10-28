#include <ipanema/integration.h>
#define FUNC(x) ((*func)(x))

ftype quad_gauss_legendre(ftype (*func)(ftype), const ftype a, const ftype b)
{
  //Returns the integral of the function func between a and b, by ten-point Gauss-Legendre inte- gration: the function is evaluated exactly ten times at interior points in the range of integration. {
  int j;
  ftype xr,xm,h,s;
  static ftype x[] = {0.0, 0.1488743389, 0.4333953941,
                      0.6794095682, 0.8650633666, 0.9739065285}; 
  static ftype w[] = {0.0, 0.2955242247, 0.2692667193,
                      0.2190863625, 0.1494513491, 0.0666713443};
  
  xm = 0.5 * (b+a); xr = 0.5 * (b-a);
  s = 0;
  for (j=1; j<=5; j++)
  {
    h = xr * x[j];
    s += w[j] * ( (*func)(xm+h) + (*func)(xm-h) );
  }

  return s *= xr; // scale
}



/*
  Compute the nth refinement of the integral of func between a and b by the trapezoidal
  rule. If n = 1, the worst estimate of the integral is returned. The error is
  reduced as ~ pow(2, n-2). It is not intended to be used by the user.
 */
ftype trapzd(ftype (*func)(ftype), ftype a, ftype b, int n)
{
  ftype x,tnm,sum,del;
  static ftype s;
  int it,j;
  if (n == 1)
  {
    return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
  }
  else
  {
    for (it=1,j=1;j<n-1;j++)
    {
      it <<= 1;
    }
    tnm = it;
    del = (b-a)/tnm;
    x = a + 0.5 * del;
    for (sum=0.0,j=1; j<=it; j++,x+=del)
    {
      sum += FUNC(x); 
      s = 0.5 * (s+(b-a)*sum/tnm);
    }
    return s;
  }
}



ftype trapz(ftype (*func)(ftype), const ftype a, const ftype b)
{
  ftype s = 0.0;
  ftype olds = 0.0;

  for (int j=1; j<=JMAX; j++)
  {
    s = trapzd(func, a, b, j);
    if (j > 5)
    {
      // Avoid spurious early convergence.
      if (fabs(s-olds) < EPS*fabs(olds) || (s == 0.0 && olds == 0.0))
      {
        return s;
      }
    }
    olds = s;
  }
  // ERROR("Too many steps in routine qtrap");
  return 0.0;
}


