import numpy as np
from uncertainties import unumpy as unp
import numdifftools as ndt


__all__ = ['uncertainty_wrapper', 'get_confidence_bands']



# this was an old function...
# def numericJacobian(f, x, vals, f_size = 1):
#   J = np.zeros([len(x), f_size, len(vals)])
#   for l in range(0,len(vals)):
#     if vals[l]!= 0:    h = np.sqrt(np.finfo(float).eps)*vals[l];
#     else:           h = 1e-14;
#     vals1 = np.copy(vals); vals1[l] += +h
#     vals2 = np.copy(vals); vals2[l] += -h;
#     f1 = f(x,*vals1).astype(np.float64)
#     f2 = f(x,*vals2).astype(np.float64)
#     thisJ = ((f(x,*vals1) - f(x,*vals2))/(2*h)).astype(np.float64)
#     J[:,0,l] = thisJ # currently only scalar f allowed
#   return J.T


def fast_jac(f, vals, f_size=1):
  """
  Computes fast numerical jacobian of a function. Be aware that this jacobian
  was self-developed to be afap.

  Parameters
  ----------
  f : callable
    Function the jacobian wants to be computed from.
  vals : list
    Points the jacobian of f wants to be computed at.
  f_size : int
    Dimension of f when evaluated at vals.

  Returns
  -------
  ndarray :
    Jacobian of f at vals
  """
  J = np.zeros([f_size, len(vals)])
  for l in range(0,len(vals)):
    if vals[l]!= 0:
      h = np.sqrt(np.finfo(float).eps)*vals[l]
    else:
      h = 1e-14
    xhp = np.copy(vals).astype(np.float64); xhp[l] += +h
    xhm = np.copy(vals).astype(np.float64); xhm[l] += -h;
    J[:,l] = (f(xhp) - f(xhm))/(2*h)
  return J if f_size > 1 else J



def fast_hesse(f, vals, f_size=1):
  """
  Computes fast numerical hessian matrix of a function

  Parameters
  ----------
  f : callable
    Function the hessian matrix wants to be computed from.
  vals : list
    Points the hessian matrix of f wants to be computed at.
  f_size : int
    Dimension of f when evaluated at vals.

  Returns
  -------
  ndarray :
    Hessian matrix of f at vals
  """
  cov = np.zeros([f_size, len(vals), len(vals)])
  g = lambda vals, pos: np.atleast_1d(f(vals))[pos]
  print(g(vals,0))
  for l in range(0,f_size):
      hessian = ndt.Hessian(g)(vals,l)
      print('\nhessian-nocorr',l,'\n',hessian)
      pos = np.where(hessian==0, 1, 0)
      rng = 1e-14 * np.random.rand(*hessian.shape)
      #hessian = np.where(pos,0,hessian)
      print('\nhessian-corr',l,'\n',hessian)
      _cov = np.where(pos, 0.0, np.linalg.inv(hessian+rng) )
      _cov = np.array([row*np.sign(row[i]) for i, row in enumerate(_cov)])
      print('\ncov',l,'\n',_cov)
      #_cov = np.linalg.inv(hessian+rng)
      _diag = np.sqrt(np.abs(_cov.diagonal()))*np.sign(_cov.diagonal())
      _diag = np.where(np.nan_to_num(_diag),np.nan_to_num(_diag),1)
      print("diag\n", _diag)
      print('\ncorrelation-1',l,'\n', _cov.T )
      print('\ncorrelation+0',l,'\n', _cov.T/_diag )
      print('\ncorrelation+1',l,'\n', (_cov.T/_diag).T )
      print('\ncorrelation+2',l,'\n', ((_cov.T/_diag).T)/_diag )
      cov[l,:,:] = ((_cov.T/_diag).T)/_diag
  return cov



def uncertainty_wrapper(func, pars, hessian=False):
  """
  Propagates the uncertanties of pars in the function f.

  Parameters
  ----------
  f : callable
    Function of pars to be propagated.
  pars : uncertainties.uarray
    Array with parameters nominal values and uncertanties which will be
    propagated in f.
  order : int (default=1)
    Whether to use linear propagation or include correlation matrix.

  Returns
  -------
  uncertainties.uarray :
    Values of f evaluated at pars with error propagation
  """
  #assert order > 0

  # ensure f is only function of pars
  f = lambda pars: func(pars)

  # get parameters and uncertainties
  try:
    _pars = list(pars.uvaluesdict().values())
    vals = np.array([_pars[k].nominal_value for k in range(0,len(pars))])
    uncs = np.array([_pars[k].std_dev for k in range(0,len(pars))])
  except:
    vals = np.array([pars[k].nominal_value for k in range(0,len(pars))])
    uncs = np.array([pars[k].std_dev for k in range(0,len(pars))])

  # compute f nominal_value
  f_val = f(vals)
  if hasattr(f_val, "__len__"):
    f_size = len(f_val)
  else:
    f_size = 1

  # get numeric derivatives
  jac = fast_jac(f, vals, f_size)
  # if order > 1:
  #   hes = fast_hesse(f, vals, f_size)

  # compute f std_dev
  f_unc = np.atleast_1d(np.zeros_like(f_val))
  for i in range(0,len(uncs)):
    f_unc[:] += jac[:,i]**2*uncs[i]**2
    if hessian:
      for j in range(0,len(uncs)):
        f_unc[:] += 2*hessian[i,j]*jac[:,i]*jac[:,j]*uncs[i]*uncs[j]

  # get uncertanties and ensure they are not negative!
  f_unc = np.sqrt(np.where(f_unc>0,f_unc,0))

  return unp.uarray(f_val,f_unc)



def get_confidence_bands(y, sigma=1):
  """
  Gets upper and lower confidence band for a given uarray y

  Parameters
  ----------
  y : uncertainties.uarray
    Array with nominal and uncertainty values
  sigma : float (default=1)
    Number of sigma the confidence band

  Returns
  -------
  tuple :
    Two-elements tuple with ndarrays defining the upper and lower confidence
    bands at asked sigma. Fist element is the upper band.
  """
  nom = unp.nominal_values(y)
  std = unp.std_devs(y)
  return nom + sigma*std, nom - sigma*std



"""
ONLY FOR DEVELOPING PURPOSES -- TESTERS
def rosen(x):
  return (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
def func(x):
  return 2*x
def func2(x):
  y = x**2
  return y - x**2
def func3(x):
  ans = np.zeros_like(x)
  for i in range(0,len(x)):
      ans[i] = np.sum(np.array([(i+1)*k*k*x[-1]*x[-1] for k in x]))
  return ans
def func4(x):
  return x[0]**4-(x[0]*x[1]*100)


# prepare some test uarray

fast_hesse(func4, [100,3], 1)


x = unp.uarray([1.,3], [0.2,0.4])
print(f"native: {func(x)} | ipanema: {uncertainty_wrapper(func,x,1)}")
print(f"native: {func(x)} | ipanema: {uncertainty_wrapper(func,x,2)}")
print(f"native: {func2(x)} | ipanema: {uncertainty_wrapper(func2,x,1)}")
print(f"native: {func2(x)} | ipanema: {uncertainty_wrapper(func2,x,2)}")

uncertainty_wrapper(rosen,x,1)
uncertainty_wrapper(rosen,x,3)
rosen(x)
uncertainty_wrapper(func3,x,1)
uncertainty_wrapper(func3,x,2)
uncertainty_wrapper(np.arctan, x, 1)
uncertainty_wrapper(np.arctan,x,2)
np.arctan(x)
"""
