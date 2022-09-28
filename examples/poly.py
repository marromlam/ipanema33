from scipy.special import sph_harm, lpmv
import matplotlib.pyplot as plt
import numpy as np
import ipanema
# from ipanema import initialize, ipanema.ristra, IPANEMALIB

ipanema.initialize("cuda", 1)

# %% prepare kernels
prog = """

#define USE_DOUBLE 1

#include <lib99ocl/core.c>
#include <lib99ocl/complex.c>
#include <lib99ocl/special.c>
#include <lib99ocl/lineshapes.c>
#include <exposed/kernels.ocl>
"""

kernels = ipanema.compile(prog, keep=False)


# %% interface


def devlpmv(m, l, x):
  xd = ipanema.ristra.allocate(x)
  out = ipanema.ristra.zeros_like(x)
  kernels.pylpmv(np.int32(m), np.int32(l), xd, out, global_size=(len(x),))
  return ipanema.ristra.get(out)


def devharms(m, l, x, y):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(y)
  out = ipanema.ristra.zeros_like(x)
  kernels.pytessel_sph_harm(
      np.int32(m), np.int32(l), xd, yd, out, global_size=(len(x),)
  )
  return ipanema.ristra.get(out)


def devcharms(m, l, x, y):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(y)
  out = ipanema.ristra.zeros_like(x).astype(np.complex128)
  kernels.pycsph_harm(np.int32(l), np.int32(
      m), xd, yd, out, global_size=(len(x),))
  return ipanema.ristra.get(out)


def harms(m, l, theta, phi):
  Y = sph_harm(abs(m), l, theta, phi)
  if m < 0:
    Y = np.sqrt(2) * (-1) ** m * Y.imag
  elif m > 0:
    Y = np.sqrt(2) * (-1) ** m * Y.real
  return np.real(Y)


print("test legendre poly")
x = np.linspace(-1, 1, 100)
for l in range(0, 5):
  for m in range(-l, l + 1):
    print(f"l,m={l:>2},{m:>2}: {np.sum(lpmv(m,l, x)-devlpmv(m,l,x))}")
# exit()
print("test spherical harmonics")
N = 100
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, N), np.linspace(-np.pi, np.pi, N))
u = np.array(u.reshape(N**2).tolist())
v = np.array(v.reshape(N**2).tolist())
for l in range(0, 5):
  for m in range(-l, l + 1):
    print(
        f"l,m={l:>2},{m:>2}: {np.sum(harms(m,l,v,u)-devharms(m,l,np.cos(u),v))}")
