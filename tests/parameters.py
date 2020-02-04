# -*- coding: utf-8 -*-

import sys
import os
sys.path.append("/home3/marcos.romero/ipanema3/")

import ipanema

import numpy as np  # Import Numpy number tools
import matplotlib.pyplot as plt
import iminuit
import json
import importlib



pars1 = ipanema.Parameters(); pars1.load('shit.json')



def residual(pars, x, y=None):
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
  return model - y


def test(*args):
  """
  test(amp,per,shift,decay)
  """
  amp = args[0]; per = args[1]; shift = args[2]; decay = args[3]
  if abs(shift) > np.pi/2: shift = shift - np.sign(shift)*np.pi
  model = amp * np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
  return ((model - y)**2).sum()


# prepare parameters
pars2 = ipanema.Parameters(); pars2.copy(pars1);





#test(pars1)

# prepare data
x = np.linspace(-5.0, 5.0, 200)
noise = np.random.normal(scale=0.0005215, size=x.size)
y = (residual(pars1, x) + noise)


test(*list(pars1.valuesdict().values()))


for key in pars2:
  pars2[key].value



def __minuit_fcn(*args):
  print(args)
  for key in pars2:
    pars2[key].value = args[ list(pars2.keys()).index(key) ]
  out = residual(pars2, x, y=None)
  return (out**2).sum()



aja = iminuit.Minuit(__minuit_fcn, forced_parameters = tuple(pars2.keys()), **someshit)

aja.migrad()




someshit = {'amp': 14.0, 'limit_amp': (13, 15),
            'period': 5.46, 'limit_period': (0, 10),
            'shift': 0.123, 'limit_shift': (0.05, 0.5),
            'decay': 0.032, 'limit_decay': (0.01, 0.05)}



#buneni = iminuit.Minuit(test, **someshit);
#buneni.migrad()

#eval("lambda %s: f(%s)" % (stringargs, stringargs))



out = ipanema.minimize(residual, method="minuit-migrad",params=pars2, args=(x,), kws={'y': y})
