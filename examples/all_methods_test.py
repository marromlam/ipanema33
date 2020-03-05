#%% Imports --------------------------------------------------------------------
#    First we should do some imports
from ipanema import initialize
initialize('python')

import os
import numpy as np
import matplotlib.pyplot as plt
import corner

from ipanema import Parameter, Parameters, Optimizer
from ipanema import all_optimize_methods


#%% Building the dataset to fit ------------------------------------------------
#    We create a ipanema.Parameters object with the true parameters.
p_true = Parameters()
a = Parameter('a', value=2.3, latex='a')
b = Parameter('b', value=5.1, latex='b')
p_true.add(a,b)



#%% Create a model -------------------------------------------------------------
#    Write down a model. Here we are writing a model that is at the same time
#    the model (it no data is provided), or the residual fcn to optimize if the
#    data is provided.

def model(pars, x, y = None):
  vals = pars.valuesdict()
  a = vals['a']
  b = vals['b']
  #model = np.sin(a*x-b)
  model = a*x-b
  if y is None:
    return model
  return (model - y)**2



#%% Prepare arrays -------------------------------------------------------------
#    It's time to create a random dataset that follows the model with p_true
#    set of parameters.

data_x = np.linspace(-2,3,50)
noise = np.random.normal(scale=3, size=data_x.size)
data_y = model(p_true,data_x) + 0.05*np.random.randn(len(data_x))
#plt.plot(data_x,data_y,'.')



#%% Fitting --------------------------------------------------------------------
#   Lets create a p_fit datset of parameters. Here we can see a way to add
#   parameters without construction a Parameter object beforehand.
p_fit = Parameters()
p_fit.add({'name':'a',   "value":2.3,  "min":2, "max":3,  'latex':r'a'})
p_fit.add({'name':'b',   "value":5.1,  "min":4, "max":6,  'latex':r'b'})

# Run the fit
model_optimizer = Optimizer(model, params=p_fit, fcn_args=(data_x,data_y), policy='raise', verbose=False)
result = {}

"""
    Estimate	Standard Error	t-Statistic	 P-Value
1	  8.04567	  0.395646	      20.3355	     9.36227*10^-37
x	  1.43589	  0.0473028	      30.3553	     2.52454*10^-51
x^2	3.9061	  0.0057577	      678.414	     4.04109*10^-180
"""

meh = model_optimizer.optimize(method=met)
meh

for met in ['powell', 'cg', 'bfgs', 'lbfgsb', 'tnc', 'cobyla', 'slsqp', 'least_squares', 'lm', 'emcee', 'minuit', 'basinhopping', 'dual_annealing', 'nelder', 'shgo']:
  print(f"Fitting with {met}...")
  result[met] = model_optimizer.optimize(method=met, verbose=False)
  result[met].params.print()
  print('\n')



result['bfgs'].hess_inv*2

from ipanema import confidence_interval
ci, fp = confidence_interval(model_optimizer, result['bfgs'])


# %% SHIT
BREAK

result['minuit']._minuit.covariance

result['bfgs']._covar_ndt
result['bfgs'].covar
result['bfgs'].hess_inv*2
result['bfgs'].hess_inv




dir(result['bfgs'])
result['bfgs']._covar_ndt
result['bfgs'].params
result['bfgs'].hess_inv
result['minuit'].params.print()
result['minuit']._minuit.matrix()
result['minuit'].cov
result['minuit'].invcov
result['lm'].params.print()
result['emcee'].params.print()
result['least_squares'].params.print()

from ipanema import confidence_interval
ci, fp = confidence_interval(model_optimizer, result['lm'])
2*(ci['a0'][0]-ci['a0'][1])
2*(ci['a1'][0]-ci['a1'][1])
2*(ci['a2'][0]-ci['a2'][-1])
ci['a2']

result['minuit'].params.print()

shit

corner.corner(result['emcee'].flatchain)
print(result['emcee'])








# %% shit
remove: 'dogleg'  'trust-ncg'
solve problems with: 'minuit' differential_evolution
result = optimize(model, method="bfgs", params=p_fit,
                  fcn_args=(data_x,), fcn_kwgs={'y': data_y}
                 )
print(result)

# Run the fit with MCMC optimization
result = optimize(model, method='emcee', params=p_fit,
                  args=(data_x,), kwgs={'y': data_y},
                  nan_policy='omit', burn=300, steps=2000, thin=20, workers=1,
                  is_weighted=False )

print(result)

plt.close()
corner.corner(result.flatchain,
    labels=['$'+p.latex+'$' for p in result.params.values()],
    smooth=True, plot_contours=True, color='C9')


# Plot the fit over the data
plt.close()
plt.plot(data_x,data_y,'k.')
plt.plot(data_x,model(result.params,data_x))
