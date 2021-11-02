# [ipanema example]: opencl

# Imports ----------------------------------------------------------------------
import ipanema
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer


# %% Prepare context -----------------------------------------------------------
#    This line will create a THREAD variable that handles the kernel compilers
#    and allocation of memort in backend if it is different from python.

ipanema.initialize('python')



#%% Create a set of ipanema parameters ----------------------------------------
#    bla bla bla
pars = ipanema.Parameters()
pars.add({'name':'mu', "value":3, 'latex':'\mu'})
pars.add({'name':'sigma', "value":5, 'latex':'\sigma'})



#%% Prepare kernel model ------------------------------------------------------
#    This should be writen in reikna syntax...

# Wrap the kerner in a python function
#@np.vectorize
def model(data, lkhd, mu, sigma):
  lkhd  = np.exp(-0.5*((data-mu)**2)/(sigma*sigma))
  lkhd *= 1/np.sqrt(2*3.1415*sigma**2)
  return lkhd

# Create cost function, in this example we will proccedd with a likelihod one
def likelihood(pars, x, prob=None):
  p = pars.valuesdict()
  if prob is None:
    prob = ipanema.ristra.allocate(0*x.get())
    return ipanema.ristra.get(model(x,prob,p['mu'],p['sigma']))
  lkhd = model(x,prob,p['mu'],p['sigma'])
  return -2*ipanema.ristra.get(ipanema.ristra.log(lkhd)) + 2*x.shape[0]


#%% Prepare arrays ------------------------------------------------------------

# Create a random variable
m, s = pars['mu'].value, pars['sigma'].value
np.random.seed(0)
x_h = np.random.normal(loc=m, scale=s, size=1000000)
pandas_host = pd.DataFrame({'x':x_h})

# Create an ipanema sample from p
sample = ipanema.Sample.from_pandas(pandas_host)

# Allocate x and prob in device
sample.allocate(x='x',prob='0*x')


#%% Fit and get the results ---------------------------------------------------

# Create an instance of ipanema.Optimizer
result = ipanema.Optimizer(likelihood,
                           params=pars,
                           fcn_args = (sample.x,sample.prob),
                           policy='filter'
                          )

# Minimize likelihood using given method
#print(result.optimize(method='emcee'))  # Markov Chain MC optimizer
print(result.optimize(method='minuit')) # Minuit optimizer (hesse)
print(result.optimize(method='bfgs'))   # scipy Broyden optimizer

#ipanema.all_optimize_methods # all ipanema optimize methods are here
