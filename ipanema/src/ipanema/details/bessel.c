WITHIN_KERNEL
ftype chebev(const ftype a, const ftype b, const ftype c[], const int m,
             const ftype x)
{
  ftype d = 0.0, dd=0.0, sv, y, y2;
  if ((x-a)*(x-b) > 0.0)
  {
    printf("x not in range in routine chebev\n");
    return 0.0;
  }
  y2 = 2.0 * ( y=(2.0*x-a-b) / (b-a) );
  for (int j=m-1; j>=1; j--) {
    sv = d; d = y2*d - dd + c[j]; dd = sv;
  }
  return y*d-dd+0.5*c[0];
}


/*
Evaluates Γ1 and Γ2 by Chebyshev expansion for |x| ≤ 1/2. Also returns 1/Γ(1 + x) and 1/Γ(1 − x). If converting to ftype precision, set NUSE1 = 7, NUSE2 = 8.
*/
WITHIN_KERNEL
void beschb(ftype x, ftype *gam1, ftype *gam2, ftype *gampl, ftype *gammi)
{
  const ftype c1[] = {
    -1.142022680371168e0, 6.5165112670737e-3, 3.087090173086e-4,
    -3.4706269649e-6, 6.9437664e-9, 3.67795e-11,-1.356e-13
  };
  const ftype c2[] = {
    1.843740587300905e0, -7.68528408447867e-2, 1.2719271366546e-3,
    -4.9717367042e-6,-3.31261198e-8, 2.423096e-10,-1.702e-13,-1.49e-15
  };

  #if USE_DOUBLE
    *gam1 = chebev(-1.0, 1.0, c1, 7, 8.0*x*x-1.0);
    *gam2 = chebev(-1.0, 1.0, c2, 8, 8.0*x*x-1.0);
  #else
    *gam1 = chebev(-1.0, 1.0, c1, 5, 8.0*x*x-1.0);
    *gam2 = chebev(-1.0, 1.0, c2, 5, 8.0*x*x-1.0);
  #endif

  *gampl = *gam2-x*(*gam1);
  *gammi = *gam2+x*(*gam1);
}



WITHIN_KERNEL
void bessel_rikv(const ftype x, const ftype xnu, ftype *ri, ftype *rk, ftype *rip, ftype *rkp)
{
  ftype a, a1, b, c, d, del, del1, delh, dels, e, f, fact, fact2, ff;
  ftype gam1, gam2, gammi, gampl, h, p, pimu, q, q1, q2, qnew;
  ftype ril, ril1, rimu, rip1, ripl,  ritemp, rk1, rkmu, rkmup, rktemp;
  ftype s, sum, sum1, x2, xi, xi2, xmu, xmu2;

  if (x <=  0.0 || xnu < 0.0)
  {
    printf("bad arguments in bessel_rikv");
  }
  const int nl = (int)(xnu+0.5);
  xmu = xnu-nl; xmu2 = xmu*xmu;
  xi = 1.0/x; xi2 = 2.0*xi; h = xnu*xi;

  if (h < 1.e-30) h = 1.e-30; b = xi2*xnu;
  d = 0.0;
  c = h;

  //nl is the number of downward re- currences of the I’s and upward recurrences of K’s. xmu lies be- tween −1/2 and 1/2.
  //Evaluate CF1 by modified Lentz’s method (§5.2).
  //Denominators cannot be zero here, so no need for special precau- tions.

  // Lentz method
  int i = 0;
  for (i=1; i<=10000; i++)
  {
    b += xi2; d = 1.0/(b+d); c = b+1.0/c; del = c*d; h = del*h;
    if (fabs(del-1.0) < MACHEP) { break; }
  }

  if (i > 10000) { printf("x too large in bessel_rikv; try asymptotic expansion"); }

  ril = 1.e-30; // initialize for recurrence
  ripl = h*ril; // initialize for recurrence
  ril1 = ril; // store
  rip1 = ripl;  // store
  fact = xnu*xi;

  for (int l = nl; l>= 1; l--)
  {
    ritemp = fact*ril+ripl; fact -=  xi; ripl = fact*ritemp+ril; ril = ritemp;
  }

  f = ripl/ril;
  if (x < 2.0)
  {
    x2 = 0.5*x; pimu = M_PI*xmu;
    fact = fabs(pimu) < MACHEP ? 1.0 : pimu/sin(pimu);
    d = -log(x2); e = xmu*d;
    fact2 = fabs(e) < MACHEP ? 1.0 : sinh(e)/e;
    beschb(xmu, &gam1, &gam2, &gampl, &gammi); // evaluation of Γ1 and Γ2.
    ff = fact*(gam1*cosh(e)+gam2*fact2*d); // f0

    sum = ff; e = exp(e); p = 0.5*e/gampl; q = 0.5/(e*gammi);
    c = 1.0; d = x2*x2; sum1 = p;
    for (i=1; i<=10000; i++)
    {
      ff = (i*ff+p+q)/(i*i-xmu2); c *= (d/i); p /= (i-xmu); q /= (i+xmu);
      del = c*ff; sum += del; del1 = c*(p-i*ff); sum1 += del1;
      if (fabs(del) < fabs(sum)*MACHEP) { break; }
    }
    if (i > 10000) { printf("bessk series failed to converge"); }
    rkmu = sum;
    rk1 = sum1*xi2;
  }
  else
  {
    b = 2.0*(1.0+x); d = 1.0/b; h = delh = d; q1 = 0.0; q2 = 1.0;
    a1 = 0.25-xmu2; q = c = a1; a = -a1; s = 1.0+q*delh;
    for (i = 2; i<=10000; i++)
    {
    // Evaluate CF2 by Steed’s algorithm (§5.2), which is OK because there can be no zero denominators.
    // Initializations for recurrence (6.7.35). First term in equation (6.7.34).
    a -= 2*(i-1); c = -a*c/i;
    qnew = (q1-b*q2)/a; q1 = q2; q2 = qnew; q += c*qnew; b += 2.0;
    d = 1.0/(b+a*d); delh = (b*d-1.0)*delh; h += delh; dels = q*delh; s += dels;
    if (fabs(dels/s) < MACHEP) { break; }
    // Need only test convergence of sum since CF2 itself converges more quickly.
    }
    if (i > 10000) { printf("bessel_rikv: failure to converge in cf2"); }
    h = a1*h;
    rkmu = sqrt(M_PI/(2.0*x))*exp(-x)/s;
    rk1 = rkmu*(xmu+x+0.5-h)*xi;
  }
  rkmup = xmu*xi*rkmu-rk1;
  rimu = xi/(f*rkmu-rkmup);
  *ri = (rimu*ril1)/ril;
  *rip = (rimu*rip1)/ril;
  for (i=1; i<=nl; i++)
  {
    rktemp = (xmu+i)*xi2*rk1+rkmu; rkmu = rk1; rk1 = rktemp;
  }
  *rk = rkmu; *rkp = xnu*xi*rkmu-rk1;
}






/*
WITHIN_KERNEL
void spherical_bessel(const int n, const float x, 
                      float *sph_j, float *sph_y, float *sph_jp, float *sph_yp)
{
  ftype order, cyl_j, cyl_jp, cyl_y, cyl_yp;
  
  if (n < 0 || x <= 0.0)
  {
    printf("bad arguments in sphbes");
  }
  const ftype order = n + 0.5;
  bessjy(x,order,&rj,&ry,&rjp,&ryp);
  // sqrt(pi/2) = 1.25331413732
  const ftype factor = 1.25331413732/sqrt(x);
  *sj = factor*rj;
  *sy = factor*ry;
  *sjp = factor*rjp - *sj*0.5/x; 
  *syp = factor*ryp - *sy*0.5/x;
}
*/
