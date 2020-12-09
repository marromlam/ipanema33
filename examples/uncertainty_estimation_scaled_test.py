DESCRIPTION = """
    This example is taken from "Tratamiento de datos físicos - Varela et al."
    where the estimators are
        a = 0.071 +/- 0.019
        b = 0.02620 +/- 0.00034
   computed as a linear regresion vai least squares fit.
   As minuit does not scale the error matrix, uncertanties coming from it are
   wrongly evaluated. Be aware of that.
"""


import numpy as np
import ipanema

# Linear model to fit data
def residual(pars, x, y=None, uy=None):
  model = pars['a'].value + pars['b'].value*x
  if y is None:
    return model
  return (1./uy**2)*(model - y)

# create a set of ipanema.Parameters
#     note c is only a number computed after minimizing a and b and it is
#     not used during the optimization
p = ipanema.Parameters()
p.add({'name':'a','value':  0.071},
      {'name':'b','value':  0.026},
      {'name':'c','value':  0.026, 'formula': '2*a+b'},
      {'name':'d','value':  0.026, 'formula': '2*a-b'})

# data
#     These numbers are directly taken from the book
x = np.array([10,20,30,40,50,60,70,80,90])
y = np.array([0.37,0.58,0.83,1.15,1.36,1.62,1.90,2.18,2.45])
uy = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05])


# optimize
#     Run with different optimizers
opt = ipanema.Optimizer(fcn_call=residual, params=p, residual_reduce='chi2',
                        fcn_kwgs={'x': x, 'y': y, 'uy': uy})
minuit = opt.optimize(method='minuit')
print("with Minuit\n", minuit.params)
bfgs = opt.optimize(method='bfgs')
print("with BFGS\n",bfgs.params)
