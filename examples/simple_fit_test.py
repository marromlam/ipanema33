#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simple fit

# Imports ----------------------------------------------------------------------
#    First we should do some imports
from ipanema import initialize
initialize('python')

import os
import numpy as np
import matplotlib.pyplot as plt
import corner

from ipanema import Parameter, Parameters, optimize



#%% Building the dataset to fit ------------------------------------------------
#    We create a ipanema.Parameters object with the true parameters.
p_true = Parameters()
amp = Parameter('amp', value=11.0, init=12, latex='A')
period = Parameter('period', value=6)
shift = Parameter('shift', value=0.1)
decay = Parameter('decay', value=0.52)
p_true.add(amp,period,shift,decay)
#print(p_true.dump_latex())



#%% Create a model -------------------------------------------------------------
#    Write down a model. Here we are writing a model that is at the same time
#    the model (it no data is provided), or the residual fcn to optimize if the
#    data is provided.

def model(pars, x, y = None):
  """Model a decaying sine wave and subtract y."""
  vals = pars.valuesdict()
  amp = vals['amp']
  per = vals['period']
  shift = vals['shift']
  decay = vals['decay']

  if abs(shift) > np.pi/2:
    shift = shift - np.sign(shift)*np.pi
  model = amp * np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
  if y is None:
    return model
  return (model - y)**2



#%% Prepare arrays -------------------------------------------------------------
#    It's time to create a random dataset that follows the model with p_true
#    set of parameters.

data_x = np.linspace(-7.0, 7.0, 100)
noise = np.random.normal(scale=0.00001, size=data_x.size)
data_y = (model(p_true, data_x) + noise)



#%% Fitting --------------------------------------------------------------------
#   Lets create a p_fit datset of parameters. Here we can see a way to add
#   parameters without construction a Parameter object beforehand.
p_fit = Parameters()
p_fit.add({'name':'amp',   "value":11,  "min":1, "max":15,  'latex':r'A'})
p_fit.add({'name':'period',"value":5,   "min":0, "max":10,  'latex':r'\tau'})
p_fit.add({'name':'shift', "value":0.1, "min":0, "max":0.5, 'latex':r'\delta'})
p_fit.add({'name':'decay', "value":0.5, "min":0, "max":1.0, 'latex':r'\Gamma'})
p_fit.print()

# Run the fit with BFGS method
result = optimize(model, method="bfgs", params=p_fit,
                  args=(data_x,), kwgs={'y': data_y}
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
