################################################################################
#                                                                              #
#                            CONFIDENCE INTERVALS                              #
#                                                                              #
################################################################################



from collections import OrderedDict
from warnings import warn

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import f



################################################################################
################################################################################

def fisher_test(ndata, nfree, new_chi, best_chi2, nfix=1):
  """
  Description Fisher test

  In:
  0.123456789:
        ndata:  Number of data points.
                int
        nfree:  Number of free parameters.
                int
     new_chi2:  Chi2 of alternate model.
                float
    best_chi2:  Best chi2.
                float
      n_fixed:  Number of fixed parameters.
                int, optional (default=1)

  Out:
            0:  Probability.
                float

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

def confidence_interval(minimizer, result, param_names=None, sigmas=[1, 2, 3],
                        tester=None, maxiter=200, verbose=False):
  """
  Calculate the confidence interval for parameters.

  A parameter scan is used to compute the probability with a given statistic,
  by default is

  In:
  0.123456789:
    minimizer:  The minimizer to use, holding objective function.
                ipanema.Optimizer
       result:  The result.
                ipanema.MinimizerResult
  param_names:  List of names of the parameters for which the confidence
                interval is computes
                list, optional (default=None, all are computed)
       sigmas:  The sigma-levels to find list.
                list, optional (default=[1, 2, 3])
      maxiter:  Maximum of iteration to find an upper limit.
                int, optional (default=200).
      verbose:  Print extra debuging information
                bool, optional (default=False)
       tester:  Function to calculate the probability from the optimized chi2.
                None or callable, optional (default=fisher_test)

  Out:
            0:  Dictionary with list of name: (sigma, values)-tuples.
                dict
            1:  Dictionary with fixed_param: ()
                dict

  """
  ci = CI(minimizer, result, param_names, tester, sigmas, verbose, maxiter)
  cl = ci.get_all_confidence_intervals()
  return cl, ci.footprints

################################################################################



################################################################################
################################################################################

class CI(object):

  def __init__(self, minimizer, result, param_names=None, tester=None,
               sigmas=[1, 2, 3], verbose=False, maxiter=50):
    self.verbose = verbose
    self.minimizer = minimizer
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
    for p in self.param_names:
      result[p] = {0:self.params[p].value}
      result[p].update( self.get_conficence_interval(p, -1) )
      result[p].update( self.get_conficence_interval(p, +1) )
    self.footprints = map_trace_to_names(self.footprints, self.params)
    return result



  def get_conficence_interval(self, param, direction):
    """
    Get the confidence interval for a parameter, in one direction.

    In:
    0.123456789:
          param:  Parameter which the confidence interval is being
                  calculated for.
                  str or ipanema.Parameter
      direction:  Left (-1) or right (+1) to look for the limit.
                  int or float

    Out:
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
    print(param,ret)
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
    self.minimizer.prepare_fit(self.params)
    out = self.minimizer.optimize(method='bfgs')
    prob = self.tester(out.ndata, out.ndata-out.nfree, out.chi2, self.best_chi2)
    x = [i.value for i in out.params.values()]
    self.footprints[param.name].append(x + [prob])
    self.params[param.name] = save_param
    return prob - offset



def confidence_interval2d(minimizer, result, x_name, y_name, nx=10, ny=10,
                    limits=None, tester=None):
    r"""Calculate confidence regions for two fixed parameters.

    The method itself is explained in *confidence_interval*: here we are fixing
    two parameters.

    Parameters
    ----------
    minimizer : Minimizer
        The minimizer to use, holding objective function.
    result : MinimizerResult
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
    >>> mini = Minimizer(some_func, params)
    >>> result = mini.leastsq()
    >>> x, y, gr = confidence_interval2d(mini, result, 'para1','para2')
    >>> plt.contour(x,y,gr)

    """
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
        """Calculate the probability."""
        if restore:
            restore_vals(org, result.params)
        x.value = vals[0]
        y.value = vals[1]
        save_x = result.params[x.name]
        save_y = result.params[y.name]
        result.params[x.name] = x
        result.params[y.name] = y
        minimizer.prepare_fit(params=result.params)
        out = minimizer.minuit()
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
