import numpy as np
import matplotlib.pyplot as plt

from ipanema import (Parameters, optimize, splot)


pars = Parameters()
pars.add(dict(name="mu", value=1))
pars.add(dict(name="sigma", value=0.2))
pars.add(dict(name="expon", value=-1/2))
pars.add(dict(name="nsig", value=0.5))
pars.add(dict(name="nbkg", value=0.5, formula="1-nsig"))


# generate some data
m_bkg = np.random.exponential(1/2, int(10e4))
m_sig = np.random.normal(1.2, 0.2, int(5e4))
pt_bkg = np.random.normal(5, 1, int(10e4))
pt_sig = np.random.normal(3, 1, int(5e4))


# apply some selection
m = np.concatenate([m_sig, m_bkg])
pt = np.concatenate([pt_sig, pt_bkg])
sel = (m > 0) & (m < 3)
m = m[sel]
pt = pt[sel]
sorter = np.argsort(m)
m = m[sorter]
pt = pt[sorter]


# model to fit: exp + gauss
def model(x, mu, sigma, expon, nsig, nbkg):
  _x = np.linspace(min(x), max(x), 1000)
  _model = nbkg * np.exp(expon*x) / np.trapz(np.exp(expon*_x), _x) 
  _model += nsig * np.exp(-((x-mu)/sigma)**2) / np.trapz(np.exp(-((_x-mu)/sigma)**2), _x)
  return _model


# define cost function: max loglikelihood
def fcn(pars, data):
  _pars = pars.valuesdict()
  _model = model(data, _pars['mu'], _pars['sigma'], _pars['expon'], _pars['nsig'], _pars['nbkg'])
  return -2*np.log(_model)


# fit
result = optimize(fcn, pars, method='minuit', fcn_kwgs=dict(data=m))
print(result)
fitpars = Parameters.clone(result.params)
fitpars['nsig'].set(value=fitpars['nsig'].value*len(m))
fitpars['nbkg'].set(value=fitpars['nbkg'].value*len(m), free=True)
fitpars = Parameters.clone(result.params)
print(fitpars)


sw = splot.compute_sweights(lambda *x, **y: model(m, *x, **y),
                                    Parameters.build(fitpars, ['mu', 'sigma', 'expon']),
                                    Parameters.build(fitpars, ['nsig', 'nbkg']))
print(sw)


plt.close()
plt.hist(m_sig, range=(0,3))
plt.hist(m, bins=30, alpha=0.3, range=(0,3))
# plt.show()
plt.savefig("meh0a.png")

plt.close()
plt.hist(pt, bins=30, alpha=0.3, range=(0,3))
# plt.show()
plt.savefig("meh0b.png")


plt.close()
plt.hist(m_sig, range=(0,3))
plt.hist(m, weights=sw['nsig'], bins=30, alpha=0.3, range=(0,3))
# plt.show()
plt.savefig("meh1.png")


plt.close()
plt.hist(pt, bins=30, alpha=0.3, range=(0,3))
plt.hist(pt_sig, bins=30, alpha=0.3, range=[0, 10])
plt.hist(pt, weights=sw['nsig'], bins=30, alpha=0.3, range=[0, 10])
plt.hist(pt_bkg, bins=30, alpha=0.3, range=[0, 10])
plt.hist(pt, weights=sw['nbkg'], bins=30, alpha=0.3, range=[0, 10])
# plt.show()
plt.savefig("meh2.png")


plt.close()
plt.plot(m, sw['nbkg'])
plt.plot(m, sw['nsig'])
# plt.show()
plt.savefig("meh3.png")






