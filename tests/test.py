
from ipanema import Parameter, Parameters
from ipanema import optimize

import matplotlib.pyplot as plt
import numpy as np



def gaussian(x,norm, mu, sigma):
  return norm * np.exp(-(x-mu)**2 / sigma)


def chi2(pars, x,y=None ):
  vals = pars.valuesdict()
  if y is not None:
    res = ( y - gaussian(x,vals['norm'],vals['mu'],vals['sigma']) )**2
  else:
    res = gaussian(x,vals['norm'],vals['mu'],vals['sigma'])
  return res


def lkhd(pars, x,y=None ):
  vals = pars.valuesdict()
  if y is not None:
    res = -2*np.log( gaussian(y,vals['norm'],vals['mu'],vals['sigma']) )
  else:
    res = gaussian(x,vals['norm'],vals['mu'],vals['sigma'])
  return res


x = np.linspace(-5,5,40)
y = gaussian(x, 3, 1, 0.4) + 0.05*np.random.randn(len(x))

plt.plot(x,y,'.')

mu = Parameter('mu',-10,latex='\mu')
sigma = Parameter('sigma',40,latex='\sigma')
norm = Parameter('norm',600,latex='norm')


params = Parameters()
params.add(mu,sigma,norm)
params

result = optimize(chi2,params,method='basinhopping',kwgs={'x':x,'y':y})

result.params

xx = np.linspace(-5,5,200)
yy = fcn(result.params, x=xx)
xxi = np.linspace(-5,5,200)
yyi = fcn(params, x=xxi)

plt.plot(x,y,'k.',xx,yy,'b-',xxi,yyi,'r--')
