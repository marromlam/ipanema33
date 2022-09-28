import ipanema
import numpy as np
from math import erf
import matplotlib.pyplot as plt

from ipanema import splot


pars = ipanema.Parameters()
pars.add(dict(name="mu", value=1))
pars.add(dict(name="sigma", value=0.2))
pars.add(dict(name="expon", value=-1 / 2))
pars.add(dict(name="nsig", value=0.5))
pars.add(dict(name="nbkg", value=0.5, formula="1-nsig"))


# generate some data
bounds = (0.0, 3.0)
nbkg = 10000
nsig = 5000

# Data and signal

np.random.seed(0)
tau = -2.0
beta = -1 / tau
bkg = np.random.exponential(beta, nbkg)
peak = np.random.normal(1.2, 0.2, nsig)
mass = np.concatenate((bkg, peak))

sig_p = np.random.normal(5, 1, size=nsig)
bck_p = np.random.normal(3, 1, size=nbkg)
p = np.concatenate([bck_p, sig_p])

sel = (mass > bounds[0]) & (mass < bounds[1])

mass = mass[sel]
p = p[sel]

sorter = np.argsort(mass)
mass = mass[sorter]
p = p[sorter]


# model to fit: exp + gauss
def model(x, mu, sigma, expon, nsig, nbkg):
  # exponential
  _epdf = np.exp(expon * x)
  _eint = (np.exp(3.0 * expon) - np.exp(0 * expon)) / expon
  # gaussian
  _gpdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
  _gint = erf(mu / (np.sqrt(2) * sigma)) - \
      erf((-3 + mu) / (np.sqrt(2) * sigma))
  _gint *= np.sqrt(np.pi / 2.0) * sigma
  return nbkg * (_epdf / _eint) + nsig * (_gpdf / _gint)


# define cost function: max loglikelihood
def fcn(pars, data):
  _pars = pars.valuesdict()
  _model = model(
      data, _pars["mu"], _pars["sigma"], _pars["expon"], _pars["nsig"], _pars["nbkg"]
  )
  return -2 * np.log(_model)


# fit
result = ipanema.optimize(
    fcn, pars, method="minuit", fcn_kwgs=dict(data=mass), tol=0.05
)
print(result)
fitpars = ipanema.Parameters.clone(result.params)
fitpars["nsig"].set(value=fitpars["nsig"].value * len(mass))
fitpars["nbkg"].set(value=fitpars["nbkg"].value * len(mass), free=True)
fitpars = ipanema.Parameters.clone(result.params)
"""
Nsig       5062     +/-      93       False
Nbkg       9912     +/- 1.2e+02       False
mean      1.195     +/-  0.0038       False
sigma       0.2     +/- 0.00021       False
lambda   -2.045     +/-   0.028       False
"""
# fitpars['nsig'].set(value=5062 / (5062+9912))
# fitpars['nbkg'].set(value=9912 / (5062+9912))
# fitpars['mu'].set(value=1.195)
# fitpars['sigma'].set(value=0.2)
# fitpars['expon'].set(value=-2.045)
print(fitpars)
print("They measure", 5062 + 9912)

_p = ipanema.Parameters.build(fitpars, ["mu", "sigma", "expon"])
_y = ipanema.Parameters.build(fitpars, ["nsig", "nbkg"])
sw = ipanema.splot.compute_sweights(
    lambda *x, **y: model(mass, *x, **y),
    ipanema.Parameters.build(fitpars, ["mu", "sigma", "expon"]),
    ipanema.Parameters.build(fitpars, ["nsig", "nbkg"]),
)
print(sw)

f, ax = plt.subplots(1, 2, figsize=(14, 7))

hist_conf = dict(bins=30, alpha=0.4, range=[0, 10])
ax[0].hist(sig_p, label="original sig p", **hist_conf)
ax[0].hist(p, weights=sw["nsig"], label="reconstructed sig p", **hist_conf)
ax[0].set_xlabel("p")
ax[0].legend(fontsize=12)

ax[1].hist(bck_p, label="original bck p", **hist_conf)
ax[1].hist(p, weights=sw["nbkg"], label="reconstructed bck p", **hist_conf)
ax[1].set_xlabel("p")
ax[1].legend(fontsize=12)
plt.show()


hist_conf = dict(bins=30, alpha=0.5, range=bounds)
plt.hist(peak, label="original sig mass", **hist_conf)
plt.hist(mass, weights=sw["nsig"], label="reconstructed sig mass", **hist_conf)
plt.xlabel("mass")
plt.legend()
plt.show()
