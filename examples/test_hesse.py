import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import oo, Symbol, integrate
import numdifftools as nft


def convolve(f, g, t, lower_limit=-oo, upper_limit=oo):
  tau = Symbol(r"\tau", real=True)
  return integrate(
      f.subs(t, tau) * g.subs(t, t - tau), (tau, lower_limit, upper_limit)
  )


def hastings(PDF, a, b, pdfmax=1, maxite=10000):
  for i in range(0, maxite):
    rand_x = (b - a) * np.random.rand(1) + a
    rand_y = pdfmax * np.random.rand(1)
    calc_y = PDF(rand_x)
    if rand_y <= calc_y:
      break
  return np.float64(rand_x)


def symgrad(f, v): return sp.Matrix([f]).jacobian(v)
def symhess(f, v): return sp.hessian(f, v)


# %% generate some events
time = np.array(
    [hastings(lambda x: np.exp(-0.4 * x), 0.3, 15) for i in range(0, 50000)]
)
plt.hist(time, 50)

# %% some coeffs
c0 = (np.random.rand(4) + 1).tolist()
c0


# %% define
t = sp.Symbol("t", real=True, positive=True)
G = sp.Symbol("\Gamma", real=True, positive=True)
sigma = sp.Symbol("\sigma", real=True, positive=True)
mu = sp.Symbol("\mu", real=True, positive=True)
c = [sp.Symbol(f"c_{i}", real=True) for i in range(0, 4)]
k = sp.Symbol("k")

# %% decay-time conv gaussian
fexp = sp.exp(-G * t)
fgauss = sp.exp(-((t) ** 2) / (2 * sigma**2)) / sp.sqrt(2 * sp.pi * sigma**2)
conv = sp.simplify(convolve(fexp, fgauss, t, lower_limit=0, upper_limit=oo))
# spline in one bin
spl = c[0] + t * (c[1] + t * (c[2] + t * c[3]))
conv


# %% both pdfs, numerical one and analytic one
lypdf = sp.lambdify((c[0], c[1], c[2], c[3], G, sigma, t), spl * conv, "numpy")


@np.vectorize
def numpdf(t, c0, c1, c2, c3, G, sigma):
  return lypdf(c0, c1, c2, c3, G, sigma, float(t))


sympdf = spl * conv


asd = np.array(
    [
        [3223.8385608, 1797.95269805, 1256.52992509, 1186.05146421],
        [1797.95269805, 1256.52992501, 1186.05146427, 1612.18429059],
        [1256.52992509, 1186.05146427, 1612.18429067, 3275.45692404],
        [1186.05146421, 1612.18429059, 3275.45692404, 9975.41214983],
    ]
)
np.corrcoef(numhesse(0))
asd = np.array(
    [
        [7.52805747, 3.81191562, 2.4276431, 2.1707403],
        [3.81191562, 2.4276431, 2.1707403, 2.97593354],
        [2.4276431, 2.1707403, 2.97593354, 6.42818206],
        [2.1707403, 2.97593354, 6.42818206, 21.09219053],
    ]
)
np.corrcoef(asd)


emp_cte = (-2 * np.sum(np.log(numpdf(time, *c0, 0.4, 0.044))) - 100) / 100
emp_cte

# %% Numerical case


def numlkhd(coeffs, emp_cte):
  return -2 * np.sum(np.log(numpdf(time, *coeffs, 0.4, 0.044)) + emp_cte)


def numhesse(cte): return nft.Hessian(lambda p: numlkhd(p, cte))(c0)
# numhesse - symhesse


numlkhd(c0, 0)
numlkhd(c0, emp_cte)

symhesse
numhesse(0)
numhesse(emp_cte)
# %% analytical
symlkhd = -2 * sp.Sum(
    sp.log(spl * conv).subs(t, sp.Indexed("t", k)), (k, 0, len(time) - 1)
)

symlkhd


aja = sp.lambdify(t, c, s)
symlkhd(time)
s
np.sum(time)

sp.summation(time[k], (k, 0, 1))
c

subsdict = {c[i]: c0[i] for i in range(len(c))}
subsdict
sp.simplify(symgrad(symlkhd, c)).subs(subsdict)

symhessexpr = sp.lambdify(t, sp.simplify(symhess(symlkhd, c)).subs(subsdict))

symhesse = symhessexpr(time)
symhesse
