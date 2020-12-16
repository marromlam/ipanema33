#include "complex.hpp"
#include "core.hpp"

#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29


WITHIN_KERNEL
ftype factorial(int n);

WITHIN_KERNEL
ctype faddeeva( ctype z);

WITHIN_KERNEL
ftype lpmv(int l, int m, ftype cos_theta);

WITHIN_KERNEL
ftype sph_harm(int l, int m, ftype cos_theta, ftype phi);
