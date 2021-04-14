#include "core.hpp"
#include "special.hpp"







/*
WITHIN_KERNEL
ftype diff_eval(ftype d, ftype l, ftype alpha, ftype beta, ftype delta){
  //Double_t sq2pi = TMath::Sqrt(2*TMath::ACos(-1));
  //Double_t cons1 = 1./sq2pi;
  ftype gamma = alpha;// TMath::Sqrt(alpha*alpha-beta*beta);
  ftype dg = delta*gamma;
  //Double_t mu_ = mu;// - delta*beta*BK(l+1,dg)/(gamma*BK(l,dg));
  //Double_t d = x-mu;
  ftype thing = delta*delta + d*d;
  ftype sqthing = sqrt(thing);
  ftype alphasq = alpha*sqthing;
  ftype no = pow(gamma/delta,l)/rkv(l,dg)*SQRT_2PI_INV;
  ftype ns1 = 0.5-l;

  return no*pow(alpha, ns1)*pow(thing, l/2. - 1.25)*(-d*alphasq*(rkv(l - 1.5, alphasq) + rkv(l + 0.5, alphasq)) + (2.*(beta*thing + d*l) - d)*rkv(ns1, alphasq))*exp(beta*d)/2.;
}
*/





WITHIN_KERNEL
ftype ipatia(const ftype x, const ftype mu, const ftype sigma,
             const ftype lambda, const ftype zeta, const ftype beta,
             const ftype a, const ftype n, const ftype a2, const ftype n2)
{
  //ftype delta, phi, A, B, k1, k2;
  //ftype logA, logk1, logcons1;
  //ftype cons1 = -2.*lambda;
  ftype alpha, delta, k1, k2, phi, b;
  const ftype d = x-mu;                          // define the running centroid
  const ftype d2 = d*d;
  const ftype aLsigma = a*sigma;                    // left tail starting point
  const ftype aRsigma = a2*sigma;                  // right tail starting point

  if (zeta != 0.0)
  {
    // WARNING: careful if zeta -> 0. You can implement a function for the
    //          ratio, but carefull again that |nu + 1 | != |nu| + 1 so you
    //          jave to deal wiht the signs
    phi = rkv(lambda+1.,zeta)/rkv(lambda,zeta);
    alpha = sqrt(zeta*phi)/sigma;
    delta = sqrt(zeta/phi)*sigma;
    // const ftype cons1 = sigma/sqrt(phi);
    // alpha = sqrt(zeta) / const1
    // delta = sqrt(zeta) * const1
    // printf("alpha: %f, delta: %f\n", alpha, delta);

    if (d < -aLsigma)
    {
      b = -aLsigma;
      goto diff_eval; //diff_eval(-aLsigma, lambda, alpha, beta, delta);
      left_tail:
        const ftype B = -aLsigma + n*k1/k2;
        const ftype A = k1*pow(B+aLsigma,n);
        return A * pow(B-d,-n);
    }
    else if (d > aRsigma)
    {
      b = aRsigma;
      goto diff_eval; //diff_eval(aRsigma, lambda, alpha, beta, delta);
      right_tail:
        const ftype B = -aRsigma - n2*k1/k2;
        const ftype A = k1*pow(B+aRsigma,n2);
        return A*pow(B+d,-n2);
    }
    else
    {
      //printf("hyperbolic_distribution(%f,%f,%f,%f,%f)\n", d,lambda,alpha,beta,delta);
      return hyperbolic_distribution(d,lambda,alpha,beta,delta);
    }
  }
  else if (lambda < 0.0)
  {
    // For z == 0 the phi ratio is much simpler (if lambda is negative).
    // Actually this function can be analytically integrated too. This integral
    // is not yet implemented in ipanema. Some 2F1 functions need to be
    // writen before.
    ftype delta2 = (lambda>=-1.0) ? sigma : sigma * sqrt(-2.0 - 2.0*lambda);
    delta2 *= delta2;
    if (delta2 == 0 ) { printf("DIVISION BY ZERO\n"); return MAXNUM; }

    if (d < -aLsigma )
    {
       const ftype fb = exp(-beta*aLsigma);             // function at boundary
       phi = 1. + aLsigma*aLsigma/delta2;
       k1 = fb*pow(phi,lambda-0.5);
       k2 = beta*k1;
       k2 -= 2.0 * fb*(lambda-0.5) * pow(phi,lambda-1.5) * aLsigma/delta2;
       const ftype B = -aLsigma + n*k1/k2;
       const ftype A = k1*pow(B+aLsigma,n);
       return A*pow(B-d,-n);
    }
    else if (d > aRsigma)
    {
       const ftype fb = exp(beta*aRsigma);              // function at boundary
       phi = 1. + aRsigma*aRsigma/delta2;
       k1 = fb*pow(phi,lambda-0.5);
       k2 = beta*k1;
       k2 += 2.0 * fb*(lambda-0.5) * pow(phi,lambda-1.5) * aRsigma/delta2;
       const ftype B = -aRsigma - n2*k1/k2;
       const ftype A = k1*pow(B+aRsigma,n2);
       return A*pow(B+d,-n2);
    }
    else
    {
     return exp(beta*d) * pow(1.0 + d2/delta2, lambda-0.5);
    }
  }
  else
  {
     printf("zeta = 0 only suported if lambda < 0, and lambda = %f\n", lambda);
     return MAXNUM;
  }

  diff_eval:
    //const ftype gamma = sqrt(alpha*alpha - beta*beta);
    const ftype gamma = alpha;           /* original ipatia implementation */
    const ftype dg = delta * gamma;
    const ftype var = delta*delta + b*b;
    const ftype sqvar = sqrt(var);
    const ftype alphasq = alpha*sqvar;
    const ftype no = pow(gamma/delta, lambda) / rkv(lambda,dg) * SQRT_2PI_INV;
    //printf("lambda, dg, no = %f, %f, %f\n", lambda, dg, rkv(lambda,dg));
    const ftype ns1 = 0.5-lambda;

    k1 = hyperbolic_distribution(b, lambda, alpha, beta, delta);

    k2  = -b*alphasq*(rkv(lambda-1.5, alphasq) + rkv(lambda+0.5, alphasq));
    //printf("k2 at 1 %f\n", k2);
    k2 += (2.0 * (beta*var + b*lambda) - b) * rkv(ns1, alphasq);
    //printf("k2 at 2 %f\n", k2);
    k2 *= no * pow(alpha, ns1);
    //printf("k2 at 3: no*alpha^ns1*k2 %f  %f  %f\n", k2, no , pow(alpha, ns1));
    k2 *= pow(var, 0.5*lambda-1.25);
    //printf("k2 at 4 %f\n", k2);
    k2 *= 0.5 * exp(beta*b);
    //printf("%f, %f, %f\n", alpha, k1, k2);
    if (d > aRsigma) { goto right_tail; }
    goto left_tail;
}



KERNEL
void py_ipatia(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *in,
               const ftype mu, const ftype sigma,
               const ftype lambda, const ftype zeta, const ftype beta,
               const ftype aL, const ftype nL, const ftype aR, const ftype nR)
{
  const int idx = get_global_id(0);
  out[idx] = ipatia(in[idx], mu, sigma, lambda, zeta, beta, aL, nL, aR, nR);
}



KERNEL
void py_hyperbolic_distribution(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *in,
               const ftype mu, const ftype lambda, const ftype alpha,
               const ftype beta, const ftype delta)
{
  const int idx = get_global_id(0);
  printf("mu, lambda, alpha, beta, delta = %lf, %lf, %lf, %lf, %lf\n", mu, lambda, alpha, beta, delta);
  out[idx] = hyperbolic_distribution(in[idx]-mu, lambda, alpha, beta, delta);
}



KERNEL
void py_rkv(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *in, const ftype n)
{
  const int idx = get_global_id(0);
  out[idx] = rkv(n, in[idx]);
}
