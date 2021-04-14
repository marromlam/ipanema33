#ifndef _COMPLEX_H_
#define _COMPLEX_H_

#include "core.h"
#include "special.h"



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// BASICS                                                                     //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype C(const ftype re, const ftype im);



WITHIN_KERNEL
ctype cpolar(const ftype re, const ftype im);



WITHIN_KERNEL
ctype cmul(const ctype z1, const ctype z2);



WITHIN_KERNEL
ctype cdiv(const ctype z1, const ctype z2);



WITHIN_KERNEL
ctype cadd(const ctype z1, const ctype z2);



WITHIN_KERNEL
ctype csub(const ctype z1, const ctype z2);



WITHIN_KERNEL
ctype cexp(const ctype z);



WITHIN_KERNEL
ctype csquare(const ctype z);



WITHIN_KERNEL
ctype cconj(const ctype z);



WITHIN_KERNEL
ftype cnorm(const ctype z);



WITHIN_KERNEL
ftype cabs(const ctype z);



WITHIN_KERNEL
ftype cre(const ctype z);



WITHIN_KERNEL
ftype cim(const ctype z);



WITHIN_KERNEL
ftype carg(const ctype z);


WITHIN_KERNEL
ctype clog(const ctype z);



WITHIN_KERNEL
ctype cpow(const ctype w, const ctype z);



WITHIN_KERNEL
ctype csqrt(const ctype w, const ctype z);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// POLYNOMIALS and friends                                                    //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype csph_harm(const int l, const int m, const ftype cosT, const ftype phi);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE COMPLEX GAMMA FAMILY                                                   //
////////////////////////////////////////////////////////////////////////////////
#include "details/complex.c"


WITHIN_KERNEL
ctype cgammaincc(ftype a, ctype z);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE COMPLEX BESSEL FAMILY                                                  //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype cjv(const ftype n, const ctype x);



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// THE FADDEVA FAMILY                                                         //
////////////////////////////////////////////////////////////////////////////////



WITHIN_KERNEL
ctype cwofz(const ctype z);



// WITHIN_KERNEL
// ctype ipanema_erfc2(const ctype z);



// WITHIN_KERNEL
// ctype ipanema_erfc(const ctype z);



WITHIN_KERNEL
ctype cerfc(const ctype x);




#endif // _COMPLEX_H_
