# %% Cell 1
import reikna.cluda as cluda
from scipy.special import sph_harm, lpmv
import numpy as np

import reikna


# %% --ksljhglksdfhg
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.text(0.5, 0.5, "Hello World!")
fig.show()

exit()


# %% use reikna

N = 256
API = cluda.ocl_api()
THREAD = API.Thread.create()


# %% prepare kernels
prog = """

WITHIN_KERNEL
double2 cnew(const double re, const double im)
{
  return COMPLEX_CTR(double2) (re,im);
}

WITHIN_KERNEL
double cre(const double2 z)
{
  return z.x;
}


WITHIN_KERNEL
double cim(const double2 z)
{
  return z.y;
}




WITHIN_KERNEL
double factorial(const int n)
{
   if(n <= 0)
    return 1.;

   double x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}


WITHIN_KERNEL
double lpmv(const int l, const int m, const double cosT)
{
    const int L = (l<0) ? abs(l)-1 : l; 
    const int M = abs(m);
    double factor = 1.0;

    if (m<0){
        factor = pow(-1.0,(double) m) * factorial(L-M) / factorial(L+M);
    }

    // shit
    if (M>l){
        return 0;
    }

    // L = 0
    if (L==0)
    {
        return 1.0;
    }
    // L = 1
    else if (L==1)
    {
        if      (M==0) { return cosT; }
        else           { return -factor*sqrt(1.0-cosT*cosT); } // OK
    }
    // L = 2
    else if (L==2)
    { 
        if      (M==0) { return  0.5*(3.*cosT*cosT - 1.); } // OK
        else if (M==1) { return -3.0*factor*cosT*sqrt(1.-cosT*cosT); } // OK
        else           { return  3.0*factor*(1.-cosT*cosT); } // OK
    }
    // L = 3
    else if (L==3)
    { 
        double sinT = sqrt(1.0-cosT*cosT);
        if      (M==0) { return   0.5*(5.*cosT*cosT*cosT - 3.*cosT); } 
        else if (M==1) { return  -1.5*factor*(5.*cosT*cosT - 1.)*sinT; } 
        else if (M==2) { return  15.0*factor*sinT*sinT*cosT; } 
        else           { return -15.0*factor*sinT*sinT*sinT; } 
    }
    // L = 4
    else if (L==4)
    { 
        double sinT = sqrt(1.0-cosT*cosT);
        if      (M==0) { return 0.125*(35.*cosT*cosT*cosT*cosT - 30.*cosT*cosT + 3.); } 
        else if (M==1) { return  -2.5*factor*(7.*cosT*cosT*cosT - 3.*cosT)*sinT; } 
        else if (M==2) { return   7.5*factor*(7.*cosT*cosT - 1.)*sinT*sinT; } 
        else if (M==3) { return -105.*factor*sinT*sinT*sinT*cosT; } 
        else           { return  105.*factor*sinT*sinT*sinT*sinT; } 
    }
    else {
      if (get_global_id(0) == 1){
        printf("chulada");
        }
        return 0;
    }

}



WITHIN_KERNEL
double2 csph_harm(const int l, const int m, const double cosT, const double phi)
{
  double ans = lpmv(l, m, cosT);
  ans *= sqrt( ((2*l+1)*factorial(l-m)) / (4*M_PI*factorial(l+m)) );
  return cnew(ans*cos(m*phi), ans*sin(m*phi));
}





WITHIN_KERNEL
double sph_harm(const int l, const int m, const double cosT, const double phi)
{
    if(m < 0)
    {
      return pow(-1.,m) * sqrt(2.) * cim( csph_harm(l, -m, cosT, phi) );
    }
    else if(m > 0)
    {
      return pow(-1.,m) * sqrt(2.) * cre( csph_harm(l,  m, cosT, phi) );
    }
    else
    {
      return sqrt( (2.*l+1.) / (4.*M_PI) ) * lpmv(l, m, cosT);
    }
}



KERNEL
void pylpmv(const int l, const int m, GLOBAL_MEM const double *cos_theta, 
            GLOBAL_MEM double *out)
{
  int idx = get_global_id(0);
  out[idx] = lpmv(l,m,cos_theta[idx]);
}

KERNEL
void pytessel_sph_harm(const int l, const int m, 
                       GLOBAL_MEM const double *cos_theta,
                       GLOBAL_MEM const double *phi, GLOBAL_MEM double *out)
{
  int idx = get_global_id(0);
  out[idx] = sph_harm(l, m, cos_theta[idx], phi[idx]);
}

KERNEL
void pysph_harm(const int l, const int m, GLOBAL_MEM const double *cosT, 
                GLOBAL_MEM const double *phi, GLOBAL_MEM double2 *out)
{
  int idx = get_global_id(0);
  out[idx] = csph_harm(l, m, cosT[idx], phi[idx]);
}

"""

kernels = THREAD.compile(prog, keep=False)


# %% interface


def devlpmv(m, l, x):
  xd = THREAD.to_device(x)
  out = THREAD.to_device(np.zeros_like(x))
  kernels.pylpmv(np.int32(l), np.int32(m), xd, out, global_size=(len(x),))
  return out.get()


def devharms(m, l, x, y):
  xd = THREAD.to_device(x)
  yd = THREAD.to_device(y)
  out = THREAD.to_device(np.zeros_like(x))
  kernels.pytessel_sph_harm(
      np.int32(l), np.int32(m), xd, yd, out, global_size=(len(x),)
  )
  return out.get()


def devcharms(m, l, x, y):
  xd = THREAD.to_device(x)
  yd = THREAD.to_device(y)
  out = THREAD.to_device(np.zeros_like(x)).astype(np.complex128)
  kernels.pycsph_harm(np.int32(l), np.int32(
      m), xd, yd, out, global_size=(len(x),))
  return out.get()


def harms(m, l, theta, phi):
  Y = sph_harm(abs(m), l, theta, phi)
  if m < 0:
    Y = np.sqrt(2) * (-1) ** m * Y.imag
  elif m > 0:
    Y = np.sqrt(2) * (-1) ** m * Y.real
  return np.real(Y)


"""
print('test legendre poly')
x = np.linspace(-1,1,100)
for l in range(0,5):
  for m in range(-l,l+1):
    print(f'l,m={l:>2},{m:>2}: {np.sum(lpmv(m,l, x)-devlpmv(m,l,x))}')
"""
print("test spherical harmonics")
N = 100
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, N), np.linspace(-np.pi, np.pi, N))
u = np.array(u.reshape(N**2).tolist())
v = np.array(v.reshape(N**2).tolist())


for l in range(0, 5):
  for m in range(-l, l + 1):
    print(
        f"l,m={l:>2},{m:>2}: {np.sum(harms(m,l,v,u)-devharms(m,l,np.cos(u),v))}")
