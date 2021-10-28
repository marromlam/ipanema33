#include "core.h"
#include "special.h"
#include "stats.h"


WITHIN_KERNEL
ftype kolmogorov_prob(ftype z)
{
   ftype fj[4] = {-2,-8,-18,-32}, r[4];
   const ftype w = 2.50662827;
   // c1 - -pi**2/8, c2 = 9*c1, c3 = 25*c1
   const ftype c1 = -1.2337005501361697;
   const ftype c2 = -11.103304951225528;
   const ftype c3 = -30.842513753404244;
 
   ftype u = fabs(z);
   ftype p;
   if (u < 0.2)
   {
      p = 1;
   }
   else if (u < 0.755)
   {
      ftype v = 1./(u*u);
      p = 1 - w * (exp(c1*v) + exp(c2*v) + exp(c3*v)) / u;
   }
   else if (u < 6.8116)
   {
      r[1] = 0; r[2] = 0; r[3] = 0;
      ftype v = u * u;
      int maxj = max(1, nearest_int(3./u));
      for (int j=0; j<maxj; j++)
      {
         r[j] = exp(fj[j]*v);
      }
      p = 2*(r[0] - r[1] +r[2] - r[3]);
   } 
   else
   {
      p = 0;
   }
   return p;
}


WITHIN_KERNEL
ftype erf_inverse(const ftype x)
{
   // returns  the inverse error function
   // x must be  <-1<x<1

   int kMaxit    = 50;
   ftype kEps   = 1e-14;
   ftype kConst = 0.8862269254527579;     // sqrt(pi)/2.0

   if(fabs(x) <= kEps) return kConst*x;

   // Newton iterations
   ftype erfi, derfi, y0,y1,dy0,dy1;
   if(fabs(x) < 1.0)
   {
      erfi  = kConst*fabs(x);
      y0    = erf(0.9*erfi);
      derfi = 0.1*erfi;
      for (int iter=0; iter<kMaxit; iter++)
      {
         y1  = 1. - erfc(erfi);
         dy1 = fabs(x) - y1;
         if (fabs(dy1) < kEps)  {if (x < 0) return -erfi; else return erfi;}
         dy0    = y1 - y0;
         derfi *= dy1/dy0;
         y0     = y1;
         erfi  += derfi;
         if(fabs(derfi/erfi) < kEps) {if (x < 0) return -erfi; else return erfi;}
      }
   }
   return 0;
}









