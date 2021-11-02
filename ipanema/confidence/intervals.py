################################################################################
#                                                                              #
#                            CONFIDENCE INTERVALS                              #
#                                                                              #
################################################################################

"""
NOT COMPLETELY FINISHED !!!
"""

from collections import OrderedDict
from warnings import warn

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import f

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..params import Parameter, Parameters
import uncertainties as unc
from uncertainties import unumpy as unp


__all__ = ['fisher_test', 'confidence_interval', 'confidence_interval2d', 'plot_conf2d']


################################################################################
################################################################################

def fisher_test(ndata, nfree, new_chi, best_chi2, nfix=1):
  """
  Description Fisher test

  Parameters
  ----------
  ndata : int
  Number of data points.
  nfree : int
  Number of free parameters.
  new_chi2 : float
  Chi2 of alternate model.
  best_chi2 : float
  Best chi2.
  n_fixed : int, optional (default=1)
  Number of fixed parameters.


  Returns
  -------
  float
  Probability.

  """
  nfree = ndata - ( nfree + nfix )
  diff_chi = new_chi / best_chi2 - 1.0
  return f.cdf(diff_chi * nfree/nfix, nfix, nfree)



def backup_vals(params):
  temp = {k:(p.value,p.stdev) for k,p in params.items()}
  return temp



def restore_vals(temp, params):
  for k in temp.keys():
    params[k].value, params[k].stdev = temp[k]



def map_trace_to_names(trace, params):
  """Map trace to parameter names."""
  out = {}
  allnames = list(params.keys()) + ['prob']
  for name in trace.keys():
    tmp_dict = {}
    tmp = np.array(trace[name])
    for para_name, values in zip(allnames, tmp.T):
      tmp_dict[para_name] = values
    out[name] = tmp_dict
  return out



from collections import namedtuple

shit = namedtuple('shit', ['value', 'prob'])

################################################################################



################################################################################
################################################################################

def confidence_interval(optimizer, result, param_names=None, sigmas=[1, 2, 3],
                        tester=None, maxiter=200, verbose=False):
  """
  Calculate the confidence interval for parameters.

  A parameter scan is used to compute the probability with a given statistic,
  by default is

  Input
  -----
  optimizer : ipanema.Optimizer
      The optimizer to use, holding objective function.
  result : ipanema.optimizerResult
      The result.
  param_names : list
      List of names of the parameters for which the confidence interval is computes
                list, optional (default=None, all are computed)
  sigmas : list, optional (default=[1, 2, 3])
      The sigma-levels to find list.
  maxiter :
      Maximum of iteration to find an upper limit.
                int, optional (default=200).
  verbose : bool, optional (default=False)
      Print extra debuging information
  tester: callable, optional (default=fisher_test)
      Function to calculate the probability from the optimized chi2.

  Returns
  -------
  dict
      Dictionary with list of name: (sigma, values)-tuples.
  dict
      Dictionary with fixed_param: ()

  """
  if optimizer.result != 'chi2':
      sigmas = [s*2 for s in sigmas]
  ci = CI(optimizer, result, param_names, tester, sigmas, verbose, maxiter)
  cl = ci.get_all_confidence_intervals()
  return cl, ci.footprints

################################################################################



################################################################################
################################################################################

class CI(object):

  def __init__(self, optimizer, result, param_names=None, tester=None,
               sigmas=[1, 2, 3], verbose=False, maxiter=50):
    self.verbose = verbose
    self.optimizer = optimizer
    self.result = result
    self.params = result.params
    self.params_ = backup_vals(self.params)
    self.best_chi2 = result.chi2

    # If no param_names, then loo all free ones
    if param_names is None:
      param_names = [i for i in self.params if self.params[i].free]
    self.param_names = param_names
    self.fit_params = [self.params[p] for p in self.param_names]

    # check that there are at least 2 true variables!
    # check that all stdevs are sensible (including not None or NaN)
    nfree = 0
    for par in self.fit_params:
      if par.free:
        nfree += 1
        if not par.stdev > 0:
          print('shit!')
          return
    if nfree < 2:
      print('At least two free parameters are required.')
      return

    if tester is None:
      self.tester = fisher_test

    self.footprints = {i: [] for i in self.param_names}
    self.maxiter = maxiter
    self.min_rel_change = 1e-5

    self.sigmas = list(sigmas); self.sigmas.sort(); self.probs = []
    for s in self.sigmas:
      if s < 1:
        self.probs.append( s )
      else:
        self.probs.append( erf(s/np.sqrt(2)) )



  def get_all_confidence_intervals(self):
    """
    Search all confidence intervals.
    """
    result = {}
    print(self.footprints)
    for p in self.param_names:
      result[p] = {0:self.params[p].value}
      result[p].update( self.get_conficence_interval(p, -1) )
      result[p].update( self.get_conficence_interval(p, +1) )
    self.footprints = map_trace_to_names(self.footprints, self.params)
    return result



  def get_conficence_interval(self, param, direction):
    """
    Get the confidence interval for a parameter, in one direction.

    Input
    -----
    param : str or ipanema.Parameter
        Parameter which the confidence interval is being
      direction:  Left (-1) or right (+1) to look for the limit.
                  int or float

    Output:
    ______
              0:  Limit of the parameter confidence interval in the given
                  direction.
    """
    # If param is str get it from Parameters object, then fix it
    if isinstance(param, str): param = self.params[param]
    param.free = False

    # function used to calculate the probability
    get_prob = lambda val, prob: self.get_prob(param, val, prob)
    x = [i.value for i in self.params.values()]
    self.footprints[param.name].append(x + [0])

    limit, max_prob = self.find_limit(param, direction)
    start_val = a_limit = float(param.value)

    ret = {}


    # removing numpy warnings for a while
    np.seterr(all='ignore'); orig_warn_settings = np.geterr()

    for i, prob in enumerate(self.probs):
      if prob > max_prob:
        val = direction*np.inf
      else:
        try:
          val = brentq(get_prob, a_limit,  limit, rtol=.5e-4, args=prob)
        except ValueError:
          self.reset_vals()
          try:
            val = brentq(get_prob, start_val, limit, rtol=.5e-4, args=prob)
          except ValueError:
            val = np.nan
      a_limit = val
      ret[direction*self.sigmas[i]] = val

    param.free = True
    self.reset_vals()
    np.seterr(**orig_warn_settings)                   # restoring numpy warnings
    #print(param,ret)
    return ret



  def reset_vals(self):
    restore_vals(self.params_, self.params)



  def find_limit(self, param, direction):
    """Find a value for a given parameter so that prob(val) > sigmas."""
    self.reset_vals()

    # determine starting step
    if param.stdev > 0 and param.stdev < abs(param.value):
      step = param.stdev
    else:
      step = max(abs(param.value) * 0.2, 0.001)
    param.free = False
    start_val = param.value

    old_prob = 0
    limit = start_val
    i = 0

    while old_prob < max(self.probs):
      i = i + 1
      limit += step * direction

      new_prob = self.get_prob(param, limit)
      rel_change = (new_prob - old_prob) / max(new_prob, old_prob, 1.e-12)
      old_prob = new_prob

      # check for convergence
      if i > self.maxiter:
          errmsg = "maxiter={} reached ".format(self.maxiter)
          errmsg += ("and prob({}={}) = {} < "
                     "max(sigmas).".format(param.name, limit, new_prob))
          warn(errmsg)
          break

      if rel_change < self.min_rel_change:
          errmsg = "rel_change={} < {} ".format(rel_change,
                                                self.min_rel_change)
          errmsg += ("at iteration {} and prob({}={}) = {} < max"
                     "(sigmas).".format(i, param.name, limit, new_prob))
          warn(errmsg)
          break

    self.reset_vals()

    return limit, new_prob



  def get_prob(self, param, val, offset=0.0, restore=False):
    """
    Calculate the probability for given value.
    """
    if restore:
      restore_vals(self.params_, self.params)
    param.value = val
    save_param = self.params[param.name]
    self.params[param.name] = param
    self.optimizer.prepare_fit(self.params)
    out = self.optimizer.optimize(method='bfgs')
    prob = self.tester(out.ndata, out.ndata-out.nfree, out.chi2, self.best_chi2)
    print(prob)
    x = [i.value for i in out.params.values()]
    self.footprints[param.name].append(x + [prob])
    self.params[param.name] = save_param
    return prob - offset



def confidence_interval2d(optimizer, result, x_name, y_name, nx=10, ny=10,
                          limits=None, tester=None, verbose=False):
    r"""Calculate confidence regions for two fixed parameters.

    The method itself is explained in *confidence_interval*: here we are fixing
    two parameters.

    Parameters
    ----------
    optimizer : optimizer
        The optimizer to use, holding objective function.
    result : optimizerResult
        The result of running minimize().
    x_name : str
        The name of the parameter which will be the x direction.
    y_name : str
        The name of the parameter which will be the y direction.
    nx : int, optional
        Number of points in the x direction.
    ny : int, optional
        Number of points in the y direction.
    limits : tuple, optional
        Should have the form ((x_upper, x_lower), (y_upper, y_lower)). If not
        given, the default is 5 std-errs in each direction.
    tester : None or callable, optional
        Function to calculate the probability from the optimized chi-square.
        Default is None and uses built-in fisher_test (i.e., F-test).

    Returns
    -------
    x : numpy.ndarray
        X-coordinates (same shape as nx).
    y : numpy.ndarray
        Y-coordinates (same shape as ny).
    grid : numpy.ndarray
        Grid containing the calculated probabilities (with shape (nx, ny)).

    Examples
    --------
    >>> mini = optimizer(some_func, params)
    >>> result = mini.leastsq()
    >>> x, y, gr = confidence_interval2d(mini, result, 'para1','para2')
    >>> plt.contour(x,y,gr)

    """
    params = Parameters.clone(result.params)
    for p in params.values():
      p.init = p.value
    #print('params after clone:', params)


    best_chi2 = result.chi2
    org = backup_vals(result.params)
    #print(org)

    if tester is None or not hasattr(tester, '__call__'):
      tester = fisher_test

    x = params[x_name]
    y = params[y_name]

    if limits is None:
      (x_upper, x_lower) = (x.value + 5 * x.stdev, x.value - 5 * x.stdev)
      (y_upper, y_lower) = (y.value + 5 * y.stdev, y.value - 5 * y.stdev)
    elif len(limits) == 2:
      (x_upper, x_lower) = limits[0]
      (y_upper, y_lower) = limits[1]

    x_points = np.linspace(x_lower, x_upper, nx)
    y_points = np.linspace(y_lower, y_upper, ny)
    grid = np.dstack(np.meshgrid(x_points, y_points))

    x.free = False
    y.free = False
    #print('params after fix:', params)

    def get_prob(vals, restore=True):
      #print(vals)
      thisparams = Parameters.clone(params)
      thisparams[x_name].value = vals[0]; thisparams[x_name].init = vals[0]
      thisparams[y_name].value = vals[1]; thisparams[y_name].init = vals[1]
      print('params after update:', thisparams)
      print(optimizer.behavior)
      optimizer.prepare_fit(params=thisparams)
      try:
        print('heyyyy')
        #out = optimizer.scalar_optimize(method='Nelder-Mead', verbose=True, maxiter=100*len(thisparams)),
        try:
          out = optimizer.minuit(method='migrad-only', verbose=False)
          if not out.isvalid:
            if verbose:
              print(vals, " > minuit failed, let's try with Levenberg-Marquardt")
            #out = optimizer.scalar_optimize(verbose=False)
            out = optimizer.scalar_optimize(method='BFGS')
            #out = optimizer.least_squares()
        except:
          if verbose:
            print(vals, " > gradient based algos fail here, let's go more robust with Nelder-Mead")
          out = optimizer.scalar_optimize(verbose=False)
      except:
        print('death')
        #out = optimizer.scalar_optimize()
        #out = optimizer.minuit(method='migrad-only', verbose=False)
      #out = optimizer.least_squares()
      #out = optimizer.scalar_optimize()
      prob = tester(out.ndata, out.ndata-out.nfree, out.chi2, best_chi2, nfix=2.)
      return prob

    out = x_points, y_points, np.apply_along_axis(get_prob, -1, grid)

    x.free, y.free = True, True
    #result.params = Parameters.clone(params)
    result.chi2 = best_chi2
    return out


"""
def confidence_interval2d(optimizer, result, x_name, y_name, nx=10, ny=10,
                    limits=None, tester=None):
    r"Calculate confidence regions for two fixed parameters.

The method itself is explained in *confidence_interval*: here we are fixing
two parameters.

Parameters
----------
optimizer: optimizer
The optimizer to use, holding objective function.
result: optimizerResult
The result of running minimize().
x_name: str
The name of the parameter which will be the x direction.
y_name: str
The name of the parameter which will be the y direction.
nx: int, optional
Number of points in the x direction.
ny: int, optional
Number of points in the y direction.
limits: tuple, optional
Should have the form((x_upper, x_lower), (y_upper, y_lower)). If not
given, the default is 5 std-errs in each direction.
tester: None or callable, optional
Function to calculate the probability from the optimized chi-square.
Default is None and uses built-in fisher_test(i.e., F-test).

Returns
-------
x: numpy.ndarray
X-coordinates(same shape as nx).
y: numpy.ndarray
Y-coordinates(same shape as ny).
grid: numpy.ndarray
Grid containing the calculated probabilities(with shape(nx, ny)).

Examples
--------
>> > mini = optimizer(some_func, params)
>> > result = mini.leastsq()
>> > x, y, gr = confidence_interval2d(mini, result, 'para1', 'para2')
>> > plt.contour(x, y, gr)

"
    params = result.params

    best_chi2 = result.chi2
    org = backup_vals(result.params)

    if tester is None or not hasattr(tester, '__call__'):
        tester = fisher_test

    x = params[x_name]
    y = params[y_name]

    if limits is None:
        (x_upper, x_lower) = (x.value + 5 * x.stdev, x.value - 5 * x.stdev)
        (y_upper, y_lower) = (y.value + 5 * y.stdev, y.value - 5 * y.stdev)
    elif len(limits) == 2:
        (x_upper, x_lower) = limits[0]
        (y_upper, y_lower) = limits[1]

    x_points = np.linspace(x_lower, x_upper, nx)
    y_points = np.linspace(y_lower, y_upper, ny)
    grid = np.dstack(np.meshgrid(x_points, y_points))

    x.free = False
    y.free = False

    def get_prob(vals, restore=False):
        "Calculate the probability."
        if restore:
            restore_vals(org, result.params)
        x.value = vals[0]
        y.value = vals[1]
        save_x = result.params[x.name]
        save_y = result.params[y.name]
        result.params[x.name] = x
        result.params[y.name] = y
        optimizer.prepare_fit(params=result.params)
        try:
          #out = optimizer.minuit(method='migrad-only')
          #out = optimizer.least_squares()
          out = optimizer.scalar_optimize()
          prob = tester(out.ndata, out.ndata - out.nfree, out.chi2,
                         best_chi2, nfix=2.)
        except:
          out = optimizer.least_squares()
          prob = tester(out.ndata, out.ndata - out.nfree, out.chi2,
                         best_chi2, nfix=2.)
        result.params[x.name] = save_x
        result.params[y.name] = save_y
        return prob

    out = x_points, y_points, np.apply_along_axis(get_prob, -1, grid)

    x.free, y.free = True, True
    restore_vals(org, result.params)
    result.chi2 = best_chi2
    return out

"""


"""
def plot_contours(mini, result, params=False, size=(20,20)):
  # look for free parameters
  if not params:
    params = list(result.params.keys())
  nfree = sum([1 if result.params[p].free else 0 for p in params])
  print(f"ipanema is about to run ({size[0]}x{size[1]})x{int(nfree*(nfree-1)/2)} fits")

  fig, axes = plt.subplots(figsize=(10*nfree//2, 10*nfree//2), ncols=nfree, nrows=nfree)#, sharex='col', sharey='row')

  for i in range(0,nfree):
    for j in range(0,nfree):
      if i<j:
        axes[i, j].axis('off')

  with tqdm(total=int(nfree+nfree*(nfree-1)/2)) as pbar:
    for i, k1 in enumerate(params):
      for j, k2 in enumerate(params):
        if i < j:
          x,y,z = confidence_interval2d(mini, result, k1, k2, size[0], size[1])
          #axes[j, i].contourf(x, y, z, np.linspace(0, 1, 11), cmap='GnBu')
          axes[j, i].contourf(x, y, z, 4, colors=['C1','C2','C3','C4','white'])
          if j+1 == nfree:
            axes[j,i].set_xlabel(f"${result.params[k1].latex}$")
          if i == 0:
            axes[j,i].set_ylabel(f"${result.params[k2].latex}$")
          axes[j,i].set_title(f'[{i},{j}]')
          pbar.update(1)
      #x,y,z = confidence_interval2d(mini, result, k1, k2, 30, 30)
      if i == nfree-1:
        axes[i,i].plot(y, np.sum(z,0))
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
        #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
      else:
        axes[i,i].plot(x,np.sum(z,1))
        #swap(axes[i,i])
      axes[i,i].set_title(f'[{i},{i}]')
      pbar.update(1)
  return fig, axes
"""


def plot_conf2d(mini, result, params=False, size=(20, 20), verbose=False):
  # look for free parameters
  fig, axes = plt.subplots(figsize=(5, 5))
  x, y, z = confidence_interval2d(mini, result, *params, *size, verbose=verbose)# IDEA:
  axes.contourf(x, y, z, [0,1-np.exp(-0.5),1-np.exp(-2.0),1-np.exp(-4.5)],
                cmap='GnBu')
                # colors=['C1','C3','C2','C4'], alpha=0.5)

  ci, _ = confidence_interval(mini, result, params)
  for i, p in enumerate(params):
    _var_lo = unc.ufloat(result.params[p].value, abs(ci[p][-1]-ci[p][0]))
    _var_hi = unc.ufloat(result.params[p].value, abs(ci[p][+1]-ci[p][0]))
    _v = f"{_var_lo:.2uL}".split('\pm')[0]
    _l = f"{_var_lo:.2uL}".split('\pm')[1]
    _u = f"{_var_hi:.2uL}".split('\pm')[1]
    _tex = result.params[p].latex
    _p = f"parab $\pm {result.params[p].unc_round[1]}$"
    if i == 0:
      axes.set_xlabel(f'${_tex} = {_v}_{{-{_l}}}^{{+{_u}}}$ ({_p})')
    else:
      axes.set_ylabel(f'${_tex} = {_v}_{{-{_l}}}^{{+{_u}}}$ ({_p})')

  return fig, axes





# def plot_contours(mini, result, params=False, size=(20, 20)):
#   # look for free parameters
#   if params:
#     _params = params
#   else:
#     _params = list(result.params.keys())
#   params = []
#   for p in _params:
#     if result.params[p].free:
#       params.append(p)
#     else:
#       print(" WARNING: ")

#   nfree = sum([1 if result.params[p].free else 0 for p in params])
#   print(
#       f"ipanema is about to run ({size[0]}x{size[1]})x{int(nfree*(nfree-1)/2)} fits\n")

#   # , sharex='col', sharey='row')
#   fig, axes = plt.subplots(
#       figsize=(10*nfree//2, 10*nfree//2), ncols=nfree, nrows=nfree)

#   for i in range(0, nfree):
#     for j in range(0, nfree):
#       if i <= j:
#         axes[i, j].axis('off')

#   with tqdm(total=int(nfree+nfree*(nfree-1)/2)) as pbar:
#     for i, k1 in enumerate(params):
#       for j, k2 in enumerate(params):
#         if i < j:
#           x, y, z = confidence_interval2d(
#               mini, result, k1, k2, size[0], size[1])
#           #axes[j, i].contourf(x, y, z, np.linspace(0, 1, 11), cmap='GnBu')
#           # , colors=['C1','C3','C2','C4'], alpha=0.5)
#           axes[j, i].contourf(
#               x, y, z, [0, 1-np.exp(-0.5), 1-np.exp(-2.0), 1-np.exp(-4.5)], cmap='GnBu')
#           if j+1 == nfree:
#             axes[j, i].set_xlabel(f"${result.params[k1].latex}$")
#           if i == 0:
#             axes[j, i].set_ylabel(f"${result.params[k2].latex}$")
#           axes[j, i].set_title(f'[{i},{j}]')
#           pbar.update(1)
#       #x,y,z = confidence_interval2d(mini, result, k1, k2, 30, 30)
#       if i == nfree-1:
#         0#axes[i, i].plot(y, np.sum(z, 0), 'k')
#         #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
#         #axes[i,i].get_shared_y_axes().remove(axes[nfree-1,j])
#       else:
#         0#axes[i, i].plot(x, np.sum(z, 1), 'k')
#         #swap(axes[i,i])

#       pbar.update(1)
#   ci, trace = confidence_interval(mini, result, params)
#   for i, k1 in enumerate(params):
#     for j, k2 in enumerate(params):
#       print(k1, k2)
#       if i <= j:
#         for color, x in enumerate(ci[k1].values()):
#           print(color, x, ci[k1][-1], ci[k1][+1])
#           #axes[j, i].axvline(x=x, color=f'k', linestyle=':', alpha=0.5)
#           _var_lo = unc.ufloat(
#               result.params[k2].value, abs(ci[k2][-1]-ci[k2][0]))
#           _var_hi = unc.ufloat(
#               result.params[k2].value, abs(ci[k2][+1]-ci[k2][0]))
#           _v = f"{_var_lo:.2uL}".split('\pm')[0]
#           _l = f"{_var_lo:.2uL}".split('\pm')[1]
#           _u = f"{_var_hi:.2uL}".split('\pm')[1]
#           _tex = result.params[k2].latex
#           _p = f"parab $\pm {result.params[k2].unc_round[1]}$"
#       if i-1 == j:
#         axes[j, i].set_title(f'${_tex} = {_v}_{{-{_l}}}^{{+{_u}}}$ ({_p})')

#   return fig, axes













"""
    WRAPPERS TO PLOT CONFIDENCE BANDS
"""



def fast_jac(f, vals, f_size=1):
  J = np.zeros([f_size, len(vals)])
  for l in range(0,len(vals)):
    if vals[l]!= 0:
      h = np.sqrt(np.finfo(float).eps)*vals[l];
    else:
      h = 1e-14;
    xhp = np.copy(vals).astype(np.float64); xhp[l] += +h
    xhm = np.copy(vals).astype(np.float64); xhm[l] += -h;
    J[:,l] = (f(xhp) - f(xhm))/(2*h)
  return J.T





def propagate_term(der,unc):
  return der**2*unc**2




def wrap_unc(func, pars, **kwgs):

  f = lambda pars: func(pars, **kwgs)

  # get parameters and uncertainties
  vals = np.array([pars[k].nominal_value  for k in range(0,len(pars))])
  uncs = np.array([pars[k].std_dev        for k in range(0,len(pars))])

  # compute f nominal_value
  f_val = f(vals)
  if hasattr(f(vals), "__len__"):
    f_size = len(f(vals))
  else:
    f_size = 1

  # get numeric derivatives
  derivatives = fast_jac(f, vals, f_size)
  f_unc = np.zeros_like(f_val)

  # compute f std_dev
  for i in range(0,len(uncs)):
    f_unc[:] += propagate_term(derivatives[i],uncs[i])[0]
  f_unc = np.sqrt(f_unc)

  return unp.uarray(f_val,f_unc)



def get_confidence_bands(y,sigma=1):
  nom = unp.nominal_values(y)
  std = unp.std_devs(y)
  # uncertainty lines (sigma confidence)
  return nom+sigma*std, nom-sigma*std
