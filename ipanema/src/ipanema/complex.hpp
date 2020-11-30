#include "core.hpp"

// BEGIN DEVICE FUNCTIONS

WITHIN_KERNEL
ctype cnew(const ftype re, const ftype im);

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

// END DEVICE FUNCTIONS




// BEGIN HOST EXPOSED FUNCTIONS

KERNEL
void pycexp(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out);

// END HOST EXPOSED FUNCTIONS