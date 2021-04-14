WITHIN_KERNEL
ctype __core_cigamc(ftype a, ctype x)
{
    // const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    // const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized
    // const ftype BIG = 4.503599627370496e15;
    // const ftype BIGINV = 2.22044604925031308085e-16;

    // Compute  x**a * exp(-x) / gamma(a)
    ctype ax = cmul( C(a,0) , clog(x) );
    ax = csub( ax, x);
    ax = csub( ax, C(lgamma(a),0) );
    if (cabs(ax) < -MAXLOG) return C(0.,0.); // underflow
    ax = cexp(ax);

    // Continued fraction implementation
    ctype y = C(1.-a, 0);
    ctype z = cadd(x,cadd(y, C(1.,0)));
    ctype c = C(0.,0.);
    ctype pkm2 = C(1.,0.);
    ctype qkm2 = x;
    ctype pkm1 = cadd(x, C(1.,0));
    ctype qkm1 = cmul(z, x);
    ctype ans = cdiv(pkm1, qkm1);
    ctype yc, pk, qk, r;
    ftype t;

    do {
        c = cadd(c ,C(1.,0.));
        y = cadd(y ,C(1.,0.));
        z = cadd(z ,C(2.,0.));
        yc = cmul(y, c);
        pk = csub( cmul(pkm1, z) , cmul(pkm2, yc) );
        qk = csub( cmul(qkm1, z) , cmul(qkm2, yc) );
        if (cabs(qk) != 0) {
            r = cdiv(pk, qk);
            t = cabs( cdiv(csub(ans, r),r) );
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (cabs(pk) > BIG) {
            pkm2 = cmul(pkm2 ,C(BIGINV,0));
            pkm1 = cmul(pkm1 ,C(BIGINV,0));
            qkm2 = cmul(qkm2 ,C(BIGINV,0));
            qkm1 = cmul(qkm1 ,C(BIGINV,0));
        }
    } while( t > MACHEP );

    return cmul(ans, ax);
}



WITHIN_KERNEL
ctype __core_cigam(ftype a, ctype x)
{
    //const ftype MACHEP = 1.11022302462515654042E-16; // IEEE 2**-53
    //const ftype MAXLOG = 7.09782712893383996843E2; // IEEE log(2**1024) denormalized

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ctype ax = cmul( C(a,0) , clog(x) );
    ax = csub( ax, x);
    ax = csub( ax, C(lgamma(a),0) );

    //printf("log(x)= %+f %+f i\\n", cre(clog(x)), cim(clog(x)) );
    //printf("ax= %+f %+f i\\n", cre(ax), cim(ax) );
    if (cabs(ax) < -MAXLOG) return C(0.,0.); // underflow
    ax = cexp(ax);

    /* power series */
    ctype r = C(a,0);
    ctype c = C(1.0,0);
    ctype ans = C(1.0,0);

    do {
        r = cadd(r, C(1.,0.) );
        c = cmul( c , cdiv(x, r) );
        ans = cadd(ans, c);
        //printf(" * %f \\n", cnorm(cdiv(c,ans)) );
    } while (cnorm(cdiv(c,ans)) > MACHEP);

    return cmul(ans, cdiv(ax,C(a,0.)) );
}



WITHIN_KERNEL
ftype __sinc(const ftype x, const ftype sinx)
{
  // return sinc(x) = sin(x)/x, given both x and sin(x)
  // [since we only use this in cases where sin(x) has already been computed]
  return fabs(x) < 1e-4 ? 1 - (0.1666666666666666666667)*x*x : sinx / x;
}



WITHIN_KERNEL
ftype __sinh_taylor(const ftype x)
{
  // sinh(x) via Taylor series, accurate to machine precision for |x| < 1e-2
  return x * (1 + (x*x) * (0.1666666666666666666667
                         + 0.00833333333333333333333 * (x*x)));
}
