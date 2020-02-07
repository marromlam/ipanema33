#%% dd
from ipanema import Parameter, Parameters
from ipanema import optimize
from ipanema.optimizers import ALL_METHODS
from ipanema import Model
import matplotlib.pyplot as plt
import numpy as np



def my_gauss(x,norm=1, mu=0, sigma=1):
  if not norm:
    norm = (2*sigma*np.pi*np.pi)**(-0.5)
  return norm * np.exp(-(x-mu)**2 / sigma)

x = np.linspace(-5,5,10)
y = my_gauss(x, 3, 1, 0.4) + 0.05*np.random.randn(len(x))

gauss_model = Model(my_gauss)
gauss_model
#dir(gauss_model)
pars = gauss_model.make_params(norm=1,mu=1,sigma=1)
gauss_model.eval(pars,x=x)


help(gauss_model.fit)
gauss_model.fit(data=y,params=pars,method='bfgs',fit_kwgs={'x':x},x=x)

#%%


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


x = np.linspace(-5,5,1000)
y = gaussian(x, 3, 1, 0.4) + 0.05*np.random.randn(len(x))

plt.plot(x,y,'.')


mu = Parameter('mu',0.5,min=1e-2,max=1e1,latex='\mu')
sigma = Parameter('sigma',0.5,min=1e-2,max=1e1,latex='\sigma')
norm = Parameter('norm',1,min=1e-2,max=1e1,latex='norm')


params = Parameters()
params.add(mu,sigma,norm)
params.print()


all_methods = list(ALL_METHODS)

for method in all_methods:
  print(f'METHOD {method}')
  result = optimize(chi2,params,method=method,kwgs={'x':x,'y':y})
  result.params.print()



result = optimize(chi2,params,method='bfgs',kwgs={'x':x,'y':y})
result



result.params

xx = np.linspace(-5,5,200)
yy = chi2(result.params, x=xx)
xxi = np.linspace(-5,5,200)
yyi = chi2(params, x=xxi)

plt.plot(x,y,'k.',xx,yy,'b-',xxi,yyi,'r--')
