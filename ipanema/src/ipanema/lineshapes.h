#ifndef _LINESHAPES_H_
#define _LINESHAPES_H_

#include "core.h"
#include "special.h"


WITHIN_KERNEL
ftype normal( const ftype x, const ftype mu, const ftype sigma );



WITHIN_KERNEL
ftype doublenormal(const ftype x, const ftype mu, const ftype sigma, 
                   const ftype dmu, const ftype dsigma, 
                   const ftype yae, const ftype res);



WITHIN_KERNEL
ftype amoroso(const ftype x, const ftype a, const ftype theta,
              const ftype alpha, const ftype beta );



WITHIN_KERNEL
ftype hyperbolic_distribution(const ftype x, const ftype lambda,
                              const ftype alpha, const ftype beta,
                              const ftype delta);



WITHIN_KERNEL
ftype ipatia(const ftype x, const ftype mu, const ftype sigma,
             const ftype lambda, const ftype zeta, const ftype beta,
             const ftype a, const ftype n, const ftype a2, const ftype n2);



WITHIN_KERNEL
ftype shoulder(const ftype x, const ftype mu, const ftype sigma, const ftype trans);



WITHIN_KERNEL
ftype argus(const ftype x, const ftype m0, const ftype c, const ftype p);



WITHIN_KERNEL
ftype physbkg(const ftype m, const ftype m0, const ftype c, const ftype s);


WITHIN_KERNEL
ftype johnson_su(const ftype x, const ftype mu, const ftype sigma, 
    const ftype gamma, const ftype delta);

WITHIN_KERNEL
ftype johan_cruijff(const ftype x, const ftype mu, const ftype sigmaL,
                    const ftype sigmaR, const ftype alphaL, const ftype alphaR);

#endif // _LINESHAPES_H_
