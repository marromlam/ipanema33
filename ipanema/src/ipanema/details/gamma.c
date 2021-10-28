WITHIN_KERNEL
ftype __core_igamc(ftype a, ftype x)
{

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
    //const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    //const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
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
