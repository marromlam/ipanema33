################################################################################
#                                                                              #
#                                 OPTIMIZERS                                   #
#                                                                              #
################################################################################


from collections import namedtuple
from copy import deepcopy
import multiprocessing
import numbers
import warnings
import math
from timeit import default_timer as timer
import numpy as np
from numpy import ndarray, ones_like, sqrt
from numpy.dual import inv
from numpy.linalg import LinAlgError
import pandas as pd
from pandas import isnull
import numdifftools as ndt

# Import methods
from iminuit import Minuit as minuit
from scipy.optimize import leastsq as levenberg_marquardt
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import basinhopping as scipy_basinhopping
from scipy.optimize import differential_evolution, least_squares
from scipy.optimize import dual_annealing as scipy_dual_annealing
from scipy.optimize import shgo as scipy_shgo
import emcee

# Scipy functions
from scipy.stats import cauchy as cauchy_dist
from scipy.stats import norm as norm_dist
from scipy.version import version as scipy_version

import six # faime falta?
import uncertainties as unc

# Ipanema modules
from .parameter import Parameter, Parameters
from .utils.print_reports import fit_report







def asteval_with_uncertainties(*vals, **kwgs):
  """
  Calculate object value, given values for variables.
  """
  _obj = kwgs.get('_obj', None)
  _pars = kwgs.get('_pars', None)
  _names = kwgs.get('_names', None)
  _asteval = _pars._asteval
  if (_obj is None or _pars is None or _names is None or
          _asteval is None or _obj._formula_ast is None):
    return 0
  for val, name in zip(vals, _names):
    _asteval.symtable[name] = val
  return _asteval.eval(_obj._formula_ast)


wrap_ueval = unc.wrap(asteval_with_uncertainties)


def eval_stdev(obj, uvars, _names, _pars):
  """
  Evaluate uncertainty and set .stdev for a parameter `obj`.
  """
  if not isinstance(obj, Parameter) or getattr(obj, '_formula_ast', None) is None:
    return
  uval = wrap_ueval(*uvars, _obj=obj, _names=_names, _pars=_pars)
  try:
    obj.stdev = uval.std_dev
  except Exception:
    obj.stdev = 0





SCIPY_METHODS = {
  'nelder':                 'Nelder-Mead',
  'powell':                 'Powell',
  'cg':                     'CG',
  'bfgs':                   'BFGS',
  'newton':                 'Newton-CG',
  'lbfgsb':                 'L-BFGS-B',
  'tnc':                    'TNC',
  'cobyla':                 'COBYLA',
  'slsqp':                  'SLSQP',
  'dogleg':                 'dogleg',
  'trust-ncg':              'trust-ncg',
  'differential_evolution': 'differential_evolution'
}

GRADIENT_METHODS = {
  'powell':                 'Powell',
  'cg':                     'CG',
  'bfgs':                   'BFGS',
  #'newton':                 'Newton-CG',
  'lbfgsb':                 'L-BFGS-B',
  'tnc':                    'TNC',
  'cobyla':                 'COBYLA',
  'slsqp':                  'SLSQP',
  #'dogleg':                 'dogleg',
  'trust-ncg':              'trust-ncg',
  'minuit':                  'minuit',
  'lm':                     'lm',
  'least_squares':          'least_squares'
}

STOCHASTIC_METHODS = {
  'emcee':                  'emcee',
  'basinhopping':           'basinhopping',
  'dual_annealing':         'dual_annealing'
}

HEURISTIC_METHODS = {
  'differential_evolution': 'differential_evolution',
  'nelder':                 'Nelder-Mead'
}

GENETIC_METHODS = {
#
}


LIPSCHIZ_METHODS = {
  'shgo': 'shgo'
}

ALL_METHODS = {}
ALL_METHODS.update(GRADIENT_METHODS)
ALL_METHODS.update(STOCHASTIC_METHODS)
ALL_METHODS.update(HEURISTIC_METHODS)
ALL_METHODS.update(GENETIC_METHODS)
ALL_METHODS.update(LIPSCHIZ_METHODS)








def _lnprior_(value, bounds):
  """
  Get a log-prior probability

  In:
  0.123456789:
        value:  Parameter's value.
                float
       bounds:  Array with (min, max) range
                np.ndarray

  Out:
            0:  Log prior probability
                float

  """
  if np.any(value > bounds[:, 1]) or np.any(value < bounds[:, 0]):
    return -np.inf
  return 0

# REVISIT
def _lnpost_(value, fcn_call, params, param_vary, bounds, fcn_args=(),
             fcn_kwgs=None, behavior='likelihood', is_weighted=True,
             policy='raise'):
  """
  Calculate the log-posterior probability.

  See the `Optimizer.emcee` method for more details.

  In:
  0.123456789:
        value:  List of varied parameters values.
                list of floats
     fcn_call:  Cost function.
                callable
       params:  All parameters.
                ipanema.Parameters
   param_vary:  List of varied parameters names.
                list of strings
       bounds:  Array with (min, max) range
                np.ndarray
     fcn_args:  Tuple with fcn arguments
                tuple, optional
     fcn_kwgs:  Dict with fcn Keyword arguments.
                dict, optional
     behavior:  Whether the fcn is 'likelihood' based or 'chi2' based.
                str, optional
  is_weighted:  Whether the residuals are already weithted or not.
                bool, optional
       policy:  Whether to raise or filter the nan value during a fit.
                str, optional (default='raise')
  Out:
            0:  Log posterior probability.

  """
  # the comparison has to be done on value and bounds. DO NOT inject value
  # values into Parameters, then compare Parameters values to the bounds.
  # Parameters values are clipped to stay within bounds.
  if np.any(value > bounds[:, 1]) or np.any(value < bounds[:, 0]):
    return -np.inf
  for name, val in zip(param_vary, value):
    params[name].value = val

  userkwgs = {}
  if fcn_kwgs is not None:
    userkwgs = fcn_kwgs

  # update the constraints
  params.update_constraints()

  # now calculate the log-likelihood
  out = fcn_call(params, *fcn_args, **userkwgs)
  out = _nan_handler_(out, policy=policy)
  lnprob = np.asarray(out).ravel()

  if lnprob.size > 1:
    if 'log_fcn' in params and not is_weighted:
      log_fcn = params['log_fcn'].value
      c = np.log(2 * np.pi) + 2 * log_fcn
      lnprob = -0.5 * np.sum((lnprob / np.exp(log_fcn)) ** 2 + c)
    else:
      lnprob = -0.5 * (lnprob * lnprob).sum()
  else:
    if behavior == 'likelihood':
      pass
    elif behavior == 'chi2':
      lnprob *= -0.5
    else:
      raise ValueError("behaviour must be either 'likelihood' or 'chi2'.")

  return lnprob



def _random_instance_(seed=None):
  """
  Set seed to a numpy.random.RandomState instance.

  In:
  0.123456789:
         seed:  Seed to np.random.RandomState.
                int or RandomState (default=None)

  Out:
            0:  Desired instance.
                np.random.RandomState instance

  """
  if seed is None or seed is np.random:
    return np.random.mtrand._rand
  if isinstance(seed, (numbers.Integral, np.integer)):
    return np.random.RandomState(seed)
  if isinstance(seed, np.random.RandomState):
    return seed
  raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                  ' instance' % seed)



def _nan_handler_(ary, policy='filter'):
  """
  Specify behaviour when an array contains numpy.nan or numpy.inf.

  In:
  0.123456789:
          ary:  Array, tipically or residuals.
                np.ndarray
       policy:  Where to raise or filter the nan value during a fit.
                string (default=raise)

  Out:
            0:  Manipulated array

  """
  if policy not in ('filter', 'raise'):
    raise ValueError("policy must be filter', or 'raise'.")

  if policy == 'filter':
    return np.nan_to_num(ary, nan=1e12, posinf=1e12, neginf=1e12)
  elif policy == 'raise':
    if np.where(np.isfinite(ary),0,1).sum():
      raise ValueError("""
        NaN Values were found in the given array. Ipanema can handle this
        kind of problems through nan_handler. Currently the policy is set
        to 'raise', change it to 'filter' in order to ipanema omit this
        non-numerical values.
        """)
    return ary

################################################################################






################################################################################
# OptimizerResult Object #######################################################

class OptimizerResult(object):
  """
  The results of a optimization.

  After running some method, a OptimizerResult object is returned by ipanema.
  This object contains a lot of attributes that here are softly described:

  Out:
       params:  The best-fit parameters resulting from the fit.
       status:  Termination status of the optimizer.
   param_vary:  Ordered list of variable parameter names used in optimization,
                and useful for understanding the values.
        covar:  Covariance matrix from minimization.
   param_init:  List of initial values for variable parameters using.
  init_values:  Dictionary of initial values for variable parameters.
         nfev:  Number of function evaluations.
      success:  Termination status of the optimizer if it's valid or not
    errorbars:  True if uncertainties were estimated, otherwise False.
      message:  Message about fit success.
        nvary:  Number of variables in fit.
        ndata:  Number of data points.
        nfree:  Degrees of freedom.
     residual:  Return value of the objective function when using the best-fit values of the parameters.
         chi2:  Squared-sum of the residual.
      chi2red:  chi2/nfree
         nll2:  Negative log-Likelihood-squared
          aic:  Akaike Information Criterion
          bic:  Bayesian Information Criterion
    flatchain:  A flatchain view of the sampling chain. [emcee method]
  """

  def __init__(self, **kws):
    for key, val in kws.items():
      setattr(self, key, val)

  @property# REVISIT
  def flatchain(self):
    """
    Show flatchain view of the sampling chain, only if emcee method was used.
    """
    if hasattr(self, 'chain'):
      if len(self.chain.shape) == 4:
        return pd.DataFrame(self.chain[0, ...].reshape((-1, self.nvary)),
                            columns=self.param_vary)
      elif len(self.chain.shape) == 3:
        return pd.DataFrame(self.chain.reshape((-1, self.nvary)),
                            columns=self.param_vary)
    else:
      return None

  def _compute_statistics_(self):
    """
    Calculate the fitting statistics.
    """
    self.nvary = len(self.param_init)
    if isinstance(self.residual, ndarray):
      self.chi2 = (self.residual).sum()
      self.ndata = len(self.residual)
      self.nfree = self.ndata - self.nvary
    else:
      print('Error when computing statistics: residual is not an array.')
      self.chi2 = self.residual
      self.ndata = 1
      self.nfree = 1
    self.chi2red = self.chi2 / max(1, self.nfree)
    self.nll2 = self.ndata * np.log(self.chi2 / self.ndata)   # -2*loglikelihood
    self.aic = self.nll2 + 2 * self.nvary         # Akaike information criterion
    self.bic = self.nll2 + np.log(self.ndata) * self.nvary  # Bayesian info crit

  def __str__(self, corr=True, min_corr=0.05):
    return fit_report(self, show_correl=corr, min_correl=min_corr, as_string=True)

################################################################################



################################################################################
# Optimizer Object #############################################################

class Optimizer(object):
  """
  A general optimizer for curve fitting and optimization.
  """

  def __init__(self, fcn_call, params,
               fcn_args=None, fcn_kwgs=None,
               model_call=None, scale_covar=True, policy='filter',
               residual_reduce='likelihood', calc_covar=True,
               **method_kwgs):
    """
    Initialize the Optimizer class.

    The objective function should return the array of residuals to be
    optimized that afterwards will be reduced to a FCN. A fcn_call function
    usually needs data, uncertainties, weights..., these can be handled under
    fcn_args and fcn_kwgs. Parameters should be passed independently through
    the params argument.

    In:
       fcn_call:  Objective function that returns the residual
                  (array, same lengh as data). This function must have the
                  signature:
                      fcn_call(params, *fcn_args, **fcn_kwgs)
                  callable
         params:  Set of paramters.
                  ipanema.parameter.Parameters
       fcn_args:  Positional arguments to pass to fcn_call.
                  tuple, optional (default=None)
       fcn_kwgs:  Keyword arguments to pass to fcn_call.
                  dict, optional (default=None)
     model_call:  Function to be called at each fit iteration. This function
                  should have the signature:
                      model_call(params, iter, resid, *fcn_args, **fcn_kwgs)
                  callable, optional (default=None)
    scale_covar:  Scale covariance matrix
                  bool, optional (default=True)
         policy:  When a NaN value if returned ipanema can handle it in two
                  ways: 'raise', a `ValueError` is raised or 'filter', the
                  NaN value is replaced by 1e12.
                  str, optional (default='raise')
residual_reduce:  Function to convert a residual array to a scalar value,
                  ipanema comes with two reductors:
                    - 'residual_sum': sum(residuals)
                    - 'residual_squared_sum': sum(residuals**2)
                  A callable can be provided so it can be used to do the
                  reduction, but the callable should TAKE ONLY ONE argument.
                  str or callable, optional (default='residual_sum')
     calc_covar:  Whether to calculate the covariance matrix or not.
                  bool, optional (default='True')
    method_kwgs:  Options to be passed tho the selected method.
                  dict, optional (default=None)

    Out:
            void

    """
    self.fcn_call = fcn_call
    self.fcn_args = fcn_args
    self.fcn_kwgs = fcn_kwgs
    if self.fcn_args is None: self.fcn_args = []
    if self.fcn_kwgs is None: self.fcn_kwgs = {}
    self.model_call = model_call

    self.miner_kwgs = {k:v for k,v in method_kwgs.items() if k!='verbose'}

    self.calc_covar = calc_covar
    self.scale_covar = scale_covar

    self.nfev = 0
    self.nfree = 0
    self.ndata = 0
    self.ier = 0
    self._abort = False
    self.success = True
    self.errorbars = False

    self.message = None
    self.lmdif_message = None
    self.chi2 = None
    self.chi2red = None
    self.covar = None
    self.residual = None
    if isinstance(residual_reduce,str):
      self.behavior = residual_reduce
    else:
      self.behavior = 'custom'
    if residual_reduce == 'chi2':
      self.residual_reduce = self._residual_squared_sum_
    if residual_reduce == 'likelihood':
      self.residual_reduce = self._residual_likelihood_
    else:
      self.residual_reduce = residual_reduce
    self.params = params
    self.policy = policy

  @property
  def values(self):
    """
    Return Parameter values in a simple dictionary.
    """
    return {name: p.value for name, p in self.result.params.items()}



  # Residual reductions ------------------------------------------------------

  def _residual_sum_(self,array):
    """
    Reduce residual array to scalar with the sum.
    """
    out = _nan_handler_(array,self.policy).sum()
    return out



  def _residual_squared_sum_(self,array):
    """
    Reduce residual array to scalar with the squared sum.
    """
    out = _nan_handler_(array*array, self.policy).sum()
    return out


  def _residual_likelihood_(self,array):
    """
    Reduce residual array to scalar with the squared sum.
    """
    out = _nan_handler_(array*array, self.policy).sum()
    return out



  # Wrappers around fcn_call -------------------------------------------------

  def _residual_(self, fvars, rebounding=True):
    """
    This is the all-method-residual-hammer. All fcn are evaluated by this
    method. Wraps arount this exist for minuit and scipy

    In:
          fvars:  Array of values of parameters
                  array
     rebounding:  Whether to apply bound transformations or not. These
                  transformations are Minuit-like ones. There is an
                  ipanema.Parameter method that hanndles them.
                  bool (default=True)

    Out:
              0:  Residuals patched by nan_handler.
                  np.ndarray

    """

    # Get parameters and set new proposals
    params = self.result.params
    if rebounding:
      for name, val in zip(self.result.param_vary, fvars):
        params[name].value = params[name].from_internal(val)
    else:
      for name, val in zip(self.result.param_vary, fvars):
        params[name].value = val

    #print(params)
    params.update_constraints()
    #print(f'params = {params} AFTER')


    # Compute model output
    out = self.fcn_call(params, *self.fcn_args, **self.fcn_kwgs)
    self.result.nfev += 1

    # Apply method
    #    In the future this will be the swicher [chi2]/[maxll fit].
    #    That means the unse inputs the model, and a string to select
    #    the kind of fit to be performed.
    # if callable(self.model_call):
    #   abort = self.model_call(params, self.result.nfev, out,
    #                           *self.fcn_args, **self.fcn_kwgs)
    #   self._abort = self._abort or abort

    # Finishing
    if self._abort:
      self.result.residual = out
      self.result.aborted  = True
      self.result.message  = "Optimization aborted by user callback."
      self.result.message += "Could not give a complete result."
      self.result.success  = False
      raise KeyboardInterrupt("Optimization aborted by user.")
    else:
      return _nan_handler_(np.asarray(out).ravel(), policy=self.policy)



  def _wrapper_minuit_(self, *fvars, reduce = True):
    """
    Residual function used for minuit methods.

    In:
          fvars:  Array of values of parameters
                  array
         reduce:  Boolean to return FCN or vector of residuals

    Out:
              0:  Residuals
                  np.ndarray
    """
    fcn = self._residual_(list(fvars), rebounding=False)
    if reduce:
      return self.residual_reduce(fcn)
    else:
      return fcn



  def _wrapper_scipy_minimize_(self, fvars, reduce = True):
    """
    Residual function used for minuit methods.

    In:
          fvars:  Array of values of parameters
                  array
         reduce:  Boolean to return FCN or vector of residuals

    Out:
              0:  Residuals
                  np.ndarray
    """
    if self.result.method in ('shgo', 'dual_annealing', 'least_squares'):
      rebounding = False
    else:
      rebounding = True
    if fvars.shape == ():
      fvars = fvars.reshape((1,))
    fcn = self._residual_(fvars, rebounding=False)
    if reduce:
      return self.residual_reduce(fcn)
    else:
      return fcn



  # Statistics calculators ---------------------------------------------------

  def _calculate_covariance_matrix_(self, fvars):
    """
    The covariance matrix.

    In:
    0.123456789:
          fvars:  Array of optimal & free values of parameters
                  array

    Out:
              0:  Covariance matrix
                  array
    """
    nfev = deepcopy(self.result.nfev) # copy this
    try:
      unbound_res_f = lambda x: self.residual_reduce(self._residual_(x, False))
      hessian = ndt.Hessian(unbound_res_f)(fvars)
      cov = 2 * np.linalg.inv(hessian)
    except (LinAlgError, ValueError):
      return None
    finally:
      self.result.nfev = nfev # paste it!
    return cov


  # maybe note necessary anymore
  def _int2ext_cov_(self, cov_int, fvars):
    """
    Transform covariance matrix to external parameter space.
    According to Minuit,
        cov_ext = np.dot(grad.T, grad) * cov_int

    In:
    0.123456789:
        cov_int:  Internal covariance matrix
                  array
          fvars:  Array of values of parameters
                  array

    Out:
              0:  External covariance matrix
                  array

    """
    g = [self.result.params[name].scale_gradient(fvars[i]) for i, name in
         enumerate(self.result.param_vary)]
    g = np.atleast_2d(g)
    cov_ext = cov_int * np.dot(g.T, g)
    return cov_ext


  # REVISIT
  def _calculate_uncertainties_correlations_(self):
    """
    Calculate parameter uncertainties and correlations.
    """
    np.seterr(all='ignore'); orig_warn_settings = np.geterr()
    self.result.errorbars = True

    scaled_cov = self.result.cov
    if self.behavior == 'chi2':
      scaled_cov *= self.result.residual.sum() / self.result.nfree


    fvar = [self.result.params[var].value for var in self.result.param_vary]
    fvar = np.atleast_1d(fvar)

    has_formula = False
    for par in self.result.params.values():
      par.stdev, par.correl = 0, None
      has_formula = has_formula or par.formula is not None

    for ivar, name in enumerate(self.result.param_vary):
      par = self.result.params[name]
      par.stdev = sqrt(scaled_cov[ivar, ivar])
      par.correl = {}
      try:
        self.result.errorbars = self.result.errorbars and (par.stdev > 0.0)
        for jvar, varn2 in enumerate(self.result.param_vary):
          if jvar != ivar:
             par.correl[varn2] = (scaled_cov[ivar, jvar] /
                                 (par.stdev * sqrt(scaled_cov[jvar, jvar])))
      except ZeroDivisionError:
        self.result.errorbars = False

    if has_formula:
      try:
        uvars = unc.correlated_values(fvar, scaled_cov)
      except (LinAlgError, ValueError):
        uvars = None

      # for uncertainties on constrained parameters, use the calculated
      # "correlated_values", evaluate the uncertainties on the constrained
      # parameters and reset the Parameters to best-fit value
      if uvars is not None:
        for par in self.result.params.values():
          eval_stdev(par, uvars, self.result.param_vary, self.result.params)
        # restore nominal values
        for v, nam in zip(uvars, self.result.param_vary):
          self.result.params[nam].value = v.nominal_value
    np.seterr(**orig_warn_settings)



  # Prepare and unprepare fit functions --------------------------------------

  def prepare_fit(self, params=None):
    """
    In:
    0.123456789:
         params:  Parameters to use.
                  ipanema.parameter.Parameters, optional
    Out:
              0:  OptimizerResult object prepared to perform fits
                  optimizers.OptimizerResult
    """

    # Build a OptimizerResult to store fitting-info
    self.result = OptimizerResult()
    result = self.result

    # Attach parameters to OptimizerResult
    if params is not None:
      self.params = params
    if isinstance(self.params, Parameters):
      result.params = Parameters()
      result.params.copy(self.params)
    elif isinstance(self.params, (list, tuple)):
      result.params = Parameters()
      for par in self.params:
        if not isinstance(par, Parameter):
          raise InputError("The provided parameters must be ipanema.Parameters objects, can do nothing.")
        else:
          result.params[par.name] = par
    elif self.params is None:
      raise InputError("I do need a set of ipanema.Parameters to fit.")

    # Check parameter atributes and consistency
    result.param_vary = []
    result.param_init = []
    result.params.update_constraints()
    result.nfev = 0
    result.errorbars = False
    result.aborted = False
    for name, par in self.result.params.items():
      par.stdev = None
      par.correl = None
      # Which parameters are defined by formula?
      if par.formula is not None:
        par.free = False
      # Which parameters are actually variables?
      if par.free:
        result.param_vary.append(name)
        result.param_init.append(par.setup_bounds())
      # Set init_value's
      par.init_value = par.value
      if par.name is None:
        par.name = name
    result.nvary = len(result.param_vary)
    result.init_values = {n: v for n, v in zip(result.param_vary, result.param_init)}

    # Set up reduce function
    if not callable(self.residual_reduce):
      self.residual_reduce = self._residual_sum_
    return result



  def unprepare_fit(self):
    pass



  def _configure_minuit_(self, pars, **kwgs):
    """
    Configure minuit things
    """
    def parameter_minuit_config(par):
      out = {par.name: par.init}
      lims = [None,None]
      if abs(par.min) != np.inf: lims[0] = par.min
      if abs(par.max) != np.inf: lims[1] = par.max
      out.update ({"limit_" + par.name: tuple(lims)})
      return out

    config = {}
    for par in pars.keys():
      if par in self.result.param_vary:
        config.update(parameter_minuit_config(pars[par]))
    config.update(kwgs)
    return config



  # Minuit method ------------------------------------------------------------

  def minuit(self, params=None, method='hesse',
             strategy=1, errordef=1, print_level=-1, pedantic=False,
             maxiter = False,
             verbose=False, **crap):
    """
    Optimization using Minuit.
    """
    result = self.prepare_fit(params=params)
    result.method = f'Minuit ({method})'
    if not maxiter:
      maxiter = 1000 * (len(result.param_init)+1)
    if verbose:
      print(f"{'method':>25} : {method}")
      print(f"{'maxiter':>25} : {maxiter}")
      print(f"{'strategy':>25} : {strategy}")
      print(f"{'errordef':>25} : {errordef}")
      print(f"{'pedantic':>25} : {pedantic}")
      print(f"{'print_level':>25} : {print_level}")
      print(f"{'non-used arguments':>25} : {crap}")
      print(f"{'params':>25} : {result.params.valuesdict()}")

    minuit_kws = {"errordef":errordef, "print_level":print_level,
                  "pedantic":pedantic}
    minuit_kws = self._configure_minuit_(result.params, **minuit_kws)
    try:
      ret = minuit(self._wrapper_minuit_,
                   forced_parameters=self.result.param_vary, **minuit_kws)
      ret.set_strategy(strategy);
      # Call migrad
      ret.migrad(ncall=maxiter)
      _counter = 1; # set a counter
      while not ret.migrad_ok() and _counter <= 20:
        if _counter == 1:
          print(f"Goddamnit! This function is not well behaved!",\
                f"Let's give it a another try.")
        ret.migrad(ncall=1000 * (len(result.param_init)+1))
        _counter += 1
      if _counter >= 20:
        print(f"Minuit cannot handle this fcn optimization.",
              f"Call other method, ipanema provides a wide variety.")
      if method == 'hesse':
        ret.hesse()
      elif method == 'minos':
        ret.minos()
    except KeyboardInterrupt:
      pass

    if not result.aborted:
      # return minuit class (you can keep optimizing, but without ipanema)
      result._minuit = ret
      # calculate fit statistics
      result.x = np.atleast_1d(ret.args)
      result.residual = self._wrapper_minuit_(*result.x, reduce = False)
      result.nfev -= 1

    result._compute_statistics_()

    # calculate the cov and estimate uncertanties/correlations
    result.cov = np.matrix(ret.matrix())
    result.invcov = np.matrix(np.linalg.inv(result.cov))
    for ivar, ipar in enumerate(self.result.param_vary):
      par = self.result.params[ipar]
      par.value = ret.values[ipar]
      par.stdev = ret.errors[ipar]
      par.correl = {}
      try:
        self.result.errorbars = self.result.errorbars and (par.stdev > 0.0)
        for jvar, varn2 in enumerate(self.result.param_vary):
          if jvar != ivar:
             par.correl[varn2] = (self.result.cov[ivar, jvar] /
                                 (par.stdev * sqrt(self.result.cov[jvar, jvar])))
      except ZeroDivisionError:
        self.result.errorbars = False
    self.result.message  = f"Fit is valid: {ret.migrad_ok()}."
    self.result.message += f"This fit has errordef = {ret.errordef} and tol = {ret.tol}."
    self.result.message += f"Current estimated distcance to minimum is {ret.edm:.4}."

    return result



  # Scipy.optimize methods handler -------------------------------------------

  def scalar_optimize(self, params=None, method='BFGS',
                      maxiter = False, hess = False, updating = 'immediate',
                      workers = 1, strategy='best1bin',
                      popsize=15, tol=0.01, mutation=(0.5, 1),
                      recombination=0.7, seed=None, callback=None,
                      disp=False, polish=True, init='latinhypercube',
                      atol=0,
                      verbose=False, **crap):
    """
    Optimization using scipy.optimize functions. For info about the Methods
    please check :scipydoc:`optimize.optimize`
    """
    result = self.prepare_fit(params=params)
    result.method = method
    if not maxiter:
      maxiter = 1000 * (len(result.param_init)+1)
    if method != 'differential_evolution':
      updating = None
      workers = None
      strategy = None
      updating = None
      workers = None
      strategy = None
      popsize = None
      tol = None
      mutation = None
      recombination = None
      seed = None
      callback = None
      disp = None
      polish = None
      init = None
      atol = None
    if method == 'differential_evolution':
      maxiter = None
      for par in params.values():
        if (par.free and not (np.isfinite(par.min) and np.isfinite(par.max))):
          raise ValueError('differential_evolution requires finite bounds.')
        diff_ev_bounds = [(-np.pi / 2., np.pi / 2.)] * len(variables)

    if verbose:
      print(f"{'method':>25} : {method}")
      print(f"{'maxiter':>25} : {maxiter}")
      print(f"{'strategy':>25} : {strategy}")
      print(f"{'hess':>25} : {hess}")
      print(f"{'updating':>25} : {updating}")
      print(f"{'workers':>25} : {workers}")
      print(f"{'popsize':>25} : {popsize}")
      print(f"{'tol':>25} : {tol}")
      print(f"{'mutation':>25} : {mutation}")
      print(f"{'recombination':>25} : {recombination}")
      print(f"{'seed':>25} : {seed}")
      print(f"{'callback':>25} : {callback}")
      print(f"{'disp':>25} : {disp}")
      print(f"{'polish':>25} : {polish}")
      print(f"{'init':>25} : {init}")
      print(f"{'atol':>25} : {atol}")
      print(f"{'non-used arguments':>25} : {crap}")
      print(f"{'params':>25} : {result.params.valuesdict()}")

    variables = result.param_init
    params = result.params
    variables = [params[par].value for par in params]

    #scpmin_kws.update(self.miner_kwgs)
    #scpmin_kws.update(method_kwgs)

    scpmin_kws = {"updating":updating, "workers":workers, "strategy":strategy,
                  "popsize":popsize, "disp":disp, "mutation":mutation,
                  "recombination":recombination, "seed":seed, "init":init,
                  "callback":callback, "polish":polish, "tol":tol, "atol":atol
                 }
    scpmin_kws = {k:v for k,v in scpmin_kws.items() if v is not None}
    if method == 'differential_evolution':
      try:
        ret = differential_evolution(self._wrapper_scipy_minimize_, diff_ev_bounds, **scpmin_kws)
      except KeyboardInterrupt:
        pass
    else:
      try:
        ret = scipy_minimize(self._wrapper_scipy_minimize_, variables, **scpmin_kws)
      except KeyboardInterrupt:
        pass

    if not result.aborted:
      if isinstance(ret, dict):
        for attr, value in ret.items():
          setattr(result, attr, value)
      else:
        for attr in dir(ret):
          if not attr.startswith('_'):
            setattr(result, attr, getattr(ret, attr))

      result.x = np.atleast_1d(result.x)
      unbound_res_f = lambda x: self.residual_reduce(self._residual_(x, False))
      result.residual = self._residual_(result.x)
      result.nfev -= 1

    result._compute_statistics_()

    # calculate the cov and estimate uncertanties/correlations
    if (not result.aborted and self.calc_covar):
      cov = self._calculate_covariance_matrix_(result.x)
      if cov is not None:
        result.cov = cov
        self._calculate_uncertainties_correlations_()

    return result



  # MCMC Hammer --------------------------------------------------------------

  def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
            ntemps=1, pos=None, reuse_sampler=False, workers=1,
            behavior='likelihood', is_weighted=True, seed=None,
            verbose=False, progress=True):
    """
    Bayesian sampling of the posterior distribution using emcee, a well known
    Markov Chain Monte Carlo package. The emcee package assumes that the
    prior is uniform. It is highly recommended to visit:
        http://dan.iel.fm/emcee/current/user/line/
    The method samples the posterior distribution of the parameters, to do so
    it needs to calculate the log-posterior probability of the model
    parameters.

    In:
         params:  Set of parameters to be used.
                  ipanema.parameter.Parameters, optional
          steps:  Number of samples to draw from the posterior distribution
                  int, optional (default=1000)
       nwalkers:  From statistics it follows nwalkers >> nvary. As it says
                  the emcee documentation:
                      "Walkers are the members of the ensemble. They are
                      almost like separate Metropolis-Hastings chains but, of
                      course, the proposal distribution for a given walker
                      depends on the positions of all the other walkers in
                      the ensemble."
                  int, optional (default=1000)
           burn:  Number of sables to be discarded from the begining of the
                  samplingint.
                  int, optional (default=0)
           thin:  mod(#samples,thin) it the number of accepted samples.
                  int, optional (default=1)
         ntemps:  Parallel Tempering if ntemps>1
                  int, optional (default=1)
            pos:  Specify the initial positions for the sampler.  If `ntemps == 1`
                  then `pos.shape` should be `(nwalkers, nvary)`. Otherwise,
                  `(ntemps, nwalkers, nvary)`. You can also initialise using a
                  previous chain that had the same `ntemps`, `nwalkers` and
                  `nvary`. Note that `nvary` may be one larger than you expect it
                  to be if your `fcn_call` returns an array and `is_weighted is
                  False`.
                  array, optional (default=None)
  reuse_sampler:  If emcee was already used to optimize a function and there
                  is no change in the parameters, then one can continue
                  drawing from the same sampler. This argument skips emcee
                  to load other arguments, so be aware.
                  bool, optional (default=False)
        workers:  For parallelization of sampling.
                  pool-like or int, optional (default=1)
       behavior:  Whether the function-call method returns a log-posterior
                  probability ('posterior') or a chi2 ('chi2')
                  str, optional (default='posterior').
    is_weighted:  If True, emcee will supose that residuals have been
                  divided by the true measurement uncertainty; if False,
                  is assumed that unweighted residuals are passed.
                  In this second case `emcee` will employ a positive
                  measurement uncertainty during the sampling. This
                  measurement uncertainty will be present in the output
                  params and output chain with the name `log_fcn`.
                  bool, optional (default=True)
           seed:  Seed for numpy random generator.
                  int or `numpy.random.RandomState`, optional (default=None)
       progress:  Flag to show a process-bar ot the sampling.
                  bool, optional (default=True)

    Out:
              0:  Optimizer result object that in general include all info
                  that the selected method provides.
    """
    tparams = params
    # if you're reusing the sampler then ntemps, nwalkers have to be
    # determined from the previous sampling
    if reuse_sampler:
      if not hasattr(self, 'sampler') or not hasattr(self, '_lastpos'):
        raise ValueError("You wanted to use an existing sampler, but "
                         "it hasn't been created yet")
      if len(self._lastpos.shape) == 2:
        ntemps = 1
        nwalkers = self._lastpos.shape[0]
      elif len(self._lastpos.shape) == 3:
        ntemps = self._lastpos.shape[0]
        nwalkers = self._lastpos.shape[1]
      tparams = None

    result = self.prepare_fit(params=tparams)
    params = result.params
    result.method = 'emcee'

    # check whether the fcn_call returns a vector of residuals
    out = self.fcn_call(params, *self.fcn_args, **self.fcn_kwgs)
    out = np.asarray(out).ravel()
    if out.size > 1 and is_weighted is False:
        # we need to marginalise over a constant data uncertainty
        if 'log_fcn' not in params:
            # log_fcn should already be in params if is_weighted was
            # previously set to True.
            params.add({'name':'log_fcn', 'value':0.01, 'min':-np.inf, 'max':np.inf, 'free':True})
            # have to re-prepare the fit
            result = self.prepare_fit(params)
            params = result.params



    # Removing internal parameter scaling. We could possibly keep it,
    # but I don't know how this affects the emcee sampling.
    bounds = []
    var_arr = np.zeros(len(result.param_vary))
    i = 0
    for par in params:
        param = params[par]
        if param.formula is not None:
            param.free = False
        if param.free:
            var_arr[i] = param.value
            i += 1
        else:
            # don't want to append bounds if they're not being varied.
            continue

        param.from_internal = lambda val: val
        lb, ub = param.min, param.max
        if lb is None or lb is np.nan:
            lb = -np.inf
        if ub is None or ub is np.nan:
            ub = np.inf
        bounds.append((lb, ub))
    bounds = np.array(bounds)

    self.nvary = len(result.param_vary)

    # set up multiprocessing options for the samplers
    auto_pool = None
    sampler_kwgs = {}
    if isinstance(workers, int) and workers > 1:
        auto_pool = multiprocessing.Pool(workers)
        sampler_kwgs['pool'] = auto_pool
    elif hasattr(workers, 'map'):
        sampler_kwgs['pool'] = workers

    # function arguments for the log-probability functions
    # these values are sent to the log-probability functions by the sampler.
    lnprob_args = (self.fcn_call, params, result.param_vary, bounds)
    lnprob_kwgs = {'is_weighted': is_weighted,
                     'behavior': behavior,
                     'fcn_args': self.fcn_args,
                     'fcn_kwgs': self.fcn_kwgs,
                     'policy': self.policy}

    if ntemps > 1:
        # the prior and likelihood function args and kwgs are the same
        sampler_kwgs['loglargs'] = lnprob_args
        sampler_kwgs['loglkwgs'] = lnprob_kwgs
        sampler_kwgs['logpargs'] = (bounds,)
    else:
        sampler_kwgs['args'] = lnprob_args
        sampler_kwgs['kwargs'] = lnprob_kwgs

    # set up the random number generator
    rng = _random_instance_(seed)

    # now initialise the samplers
    if reuse_sampler:
        if auto_pool is not None:
            self.sampler.pool = auto_pool

        p0 = self._lastpos
        if p0.shape[-1] != self.nvary:
            raise ValueError("You cannot reuse the sampler if the number"
                             "of freeing parameters has changed")
    elif ntemps > 1:
        # Parallel Tempering
        # jitter the starting position by scaled Gaussian noise
        p0 = 1 + rng.randn(ntemps, nwalkers, self.nvary) * 1.e-4
        p0 *= var_arr
        self.sampler = emcee.PTSampler(ntemps, nwalkers, self.nvary,
                                       _lnpost_, _lnprior_, **sampler_kwgs)
    else:
        p0 = 1 + rng.randn(nwalkers, self.nvary) * 1.e-4
        p0 *= var_arr
        self.sampler = emcee.EnsembleSampler(nwalkers, self.nvary,
                                             _lnpost_, **sampler_kwgs)

    # user supplies an initialisation position for the chain
    # If you try to run the sampler with p0 of a wrong size then you'll get
    # a ValueError. Note, you can't initialise with a position if you are
    # reusing the sampler.
    if pos is not None and not reuse_sampler:
        tpos = np.asfarray(pos)
        if p0.shape == tpos.shape:
            pass
        # trying to initialise with a previous chain
        elif tpos.shape[0::2] == (nwalkers, self.nvary):
            tpos = tpos[:, -1, :]
        # initialising with a PTsampler chain.
        elif ntemps > 1 and tpos.ndim == 4:
            tpos_shape = list(tpos.shape)
            tpos_shape.pop(2)
            if tpos_shape == (ntemps, nwalkers, self.nvary):
                tpos = tpos[..., -1, :]
        else:
            raise ValueError('pos should have shape (nwalkers, nvary)'
                             'or (ntemps, nwalkers, nvary) if ntemps > 1')
        p0 = tpos

    # if you specified a seed then you also need to seed the sampler
    if seed is not None:
        self.sampler.random_state = rng.get_state()

    # now do a production run, sampling all the time
    output = self.sampler.run_mcmc(p0, steps, progress=progress)
    self._lastpos = output.coords

    # discard the burn samples and thin
    chain = self.sampler.chain[..., burn::thin, :]
    lnprobability = self.sampler.lnprobability[..., burn::thin]

    # take the zero'th PTsampler temperature for the parameter estimators
    if ntemps > 1:
        flatchain = chain[0, ...].reshape((-1, self.nvary))
    else:
        flatchain = chain.reshape((-1, self.nvary))

    quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)

    for i, var_name in enumerate(result.param_vary):
        std_l, median, std_u = quantiles[:, i]
        params[var_name].value = median
        params[var_name].stdev = 0.5 * (std_u - std_l)
        params[var_name].correl = {}

    params.update_constraints()

    # work out correlation coefficients
    corrcoefs = np.corrcoef(flatchain.T)

    for i, var_name in enumerate(result.param_vary):
        for j, var_name2 in enumerate(result.param_vary):
            if i != j:
                result.params[var_name].correl[var_name2] = corrcoefs[i, j]

    result.chain = np.copy(chain)
    result.lnprob = np.copy(lnprobability)
    result.errorbars = True
    result.nvary = len(result.param_vary)
    result.nfev = ntemps*nwalkers*steps

    # Calculate the residual with the "best fit" parameters
    out = self.fcn_call(params, *self.fcn_args, **self.fcn_kwgs)
    result.residual = _nan_handler_(out, policy=self.policy)

    # If uncertainty was automatically estimated, weight the residual properly
    if (not is_weighted) and (result.residual.size > 1):
        if 'log_fcn' in params:
            result.residual = result.residual/np.exp(params['log_fcn'].value)

    # Calculate statistics for the two standard cases:
    if isinstance(result.residual, ndarray) or (behavior == 'chi2'):
      result._compute_statistics_()

    # Handle special case unique to emcee:
    # This should eventually be moved into result._compute_statistics_.
    elif behavior == 'likelihood':
        result.ndata = 1
        result.nfree = 1

        # assuming prior prob = 1, this is true
        nll2 = -2*result.residual

        # assumes that residual is properly weighted
        result.chi2 = np.exp(nll2)

        result.chi2red = result.chi2 / result.nfree
        result.aic = nll2 + 2 * result.nvary
        result.bic = nll2 + np.log(result.ndata) * result.nvary

    if auto_pool is not None:
        auto_pool.terminate()

    return result



  # Trust-Region and Levenberg-Marquardt method ------------------------------

  def least_squares(self, params=None, method='lm', verbose=False, **method_kwgs):
    """
    Standard least squares fitting method calling a Trust Region Reflective
    algorithm, which solver the standard least squares problem or the
    Levenberg-Marquardt minimization method. This is handled by
    scipy.optimize.least_squares function.

    In:
         params:  Set of parameters, must be ipanema.parameter.Parameters
    method_kwgs:  Keyword-arguments passed to the minimization algorithm.

    Out:
              0:  Optimizer result object that in general include all info
                  that the selected method provides.

    """
    result = self.prepare_fit(params)
    if method == 'least_squares':
      result.method = 'Least-Squares'
      method_ = 'trf'
    elif method == 'lm':
      result.method = 'Levenberg-Marquardt'
      method_ = 'lm'

    replace_none = lambda x, sign: sign*np.inf if x is None else x

    start_vals, lower_bounds, upper_bounds = [], [], []

    for vname in result.param_vary:
      par = self.params[vname]
      start_vals.append(par.value)
      lower_bounds.append(replace_none(par.min, -1))
      upper_bounds.append(replace_none(par.max, +1))

    print(method_kwgs)
    try:
      if method_ == 'trf':
        ret = least_squares(lambda x: self._residual_(x,False), start_vals,
                            bounds=(lower_bounds, upper_bounds),
                            method='trf',**method_kwgs)
      elif method_ == 'lm':
        ret = least_squares(lambda x: self._residual_(x,False), start_vals,
                            method='lm',**method_kwgs)
      result.residual = ret.fun
    except KeyboardInterrupt:
      pass

    if not result.aborted:
      result.message = ret.message
      result.residual = self._residual_(ret.x, False)
      result.nfev -= 1

    # Fit statistics
    result._compute_statistics_()

    # Uncertanties and correlations
    if (not result.aborted and self.calc_covar):
      _covar_ndt = self._calculate_covariance_matrix_(ret.x)
      if _covar_ndt is not None:
        result.cov = self._int2ext_cov_(_covar_ndt, ret.x)
        self._calculate_uncertainties_correlations_()

    return result


  def levenberg_marquardt(self, params=None, verbose=False, **kws):
        """
        merge this with least squares...
        """
        result = self.prepare_fit(params=params)
        result.method = 'Levenberg-Marquardt (lm)'
        result.nfev -= 2  # correct for "pre-fit" initialization/checks
        variables = result.param_init
        nvars = len(variables)
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7, col_deriv=False,
                     gtol=1.e-7, maxfev=2000*(nvars+1))

        lskws.update(self.miner_kwgs)
        lskws.update(kws)


        self.col_deriv = False

        # suppress runtime warnings during fit and error analysis
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')
        try:
          lsout = levenberg_marquardt(self._residual_, variables, **lskws)
        except KeyboardInterrupt:
          pass

        if not result.aborted:
          _best, _cov, infodict, errmsg, ier = lsout
          result.residual = self._residual_(_best)
          result.nfev -= 1
        result._compute_statistics_()

        if result.aborted:
          return result

        result.ier = ier
        result.lmdif_message = errmsg
        result.success = ier in [1, 2, 3, 4]
        if ier in {1, 2, 3}:
            result.message = 'Fit succeeded.'
        elif ier == 0:
            result.message = ('Invalid Input Parameters. I.e. more variables '
                              'than data points given, tolerance < 0.0, or '
                              'no data provided.')
        elif ier == 4:
            result.message = 'One or more variable did not affect the fit.'
        elif ier == 5:
            result.message = self._err_maxfev % lskws['maxfev']
        else:
            result.message = 'Tolerance seems to be too small.'

        # self.errorbars = error bars were successfully estimated
        result.errorbars = (_cov is not None)
        if result.errorbars:
            # transform the covariance matrix to "external" parameter space
            result.cov = self._int2ext_cov_(_cov, _best)
            # calculate parameter uncertainties and correlations
            self._calculate_uncertainties_correlations_()
        else:
            result.message = '%s Could not estimate error-bars.' % result.message

        np.seterr(**orig_warn_settings)

        return result







  # Basin - Hopping method -----------------------------------------------------

  def basinhopping(self, params=None, verbose=False, **method_kwgs):
    """
    shit shit shit
    """
    result = self.prepare_fit(params=params)
    result.method = 'basinhopping'

    basinhopping_kwgs = dict(
                              niter=100,
                              T=1.0,
                              stepsize=0.5,
                              take_step=None,
                              accept_test=None,
                              callback=None,
                              interval=50,
                              disp=False,
                              niter_success=None,
                              seed=None
                            )

    #basinhopping_kwgs.update(self.miner_kwgs)
    print(self.miner_kwgs)
    print(method_kwgs)
    basinhopping_kwgs.update(method_kwgs)

    x0 = result.param_init

    try:
      ret = scipy_basinhopping(self._wrapper_scipy_minimize_, x0,
                               **basinhopping_kwgs)
    except KeyboardInterrupt:
      pass

    if not result.aborted:
      result.message = ret.message
      result.residual = self._residual_(ret.x)
      result.nfev -= 1

    # Fit statistics
    result._compute_statistics_()

    # Uncertanties and correlations
    if (not result.aborted and self.calc_covar):
      _covar_ndt = self._calculate_covariance_matrix_(ret.x)
      if _covar_ndt is not None:
        result.cov = self._int2ext_cov_(_covar_ndt, ret.x)
        self._calculate_uncertainties_correlations_()

    return result



  # Simplicial Homology Global Optimization method ---------------------------

  def shgo(self, params=None, verbose=False, **method_kwgs):
    """
    shit shit
    """
    result = self.prepare_fit(params=params)
    result.method = 'shgo'

    shgo_kwgs = dict(
                      constraints=None,
                      n=100,
                      iters=1,
                      callback=None,
                      minimizer_kwargs=None,
                      options=None,
                      sampling_method='simplicial'
                    )

    shgo_kwgs.update(self.miner_kwgs)
    shgo_kwgs.update(method_kwgs)

    freeing = np.asarray([par.free for par in self.params.values()])
    bounds = np.asarray([(par.min, par.max) for par in
                         self.params.values()])[freeing]

    try:
      ret = scipy_shgo(self._wrapper_scipy_minimize_, bounds, **shgo_kwgs)
    except KeyboardInterrupt:
      pass

    if not result.aborted:
      for attr, value in ret.items():
        if attr in ['success', 'message']:
          setattr(result, attr, value)
        else:
          setattr(result, 'shgo_{}'.format(attr), value)
      result.residual = self._residual_(result.shgo_x, False)
      result.nfev -= 1

    # Fit statistics
    result._compute_statistics_()

    # Uncertanties and correlations
    if (not result.aborted and self.calc_covar):
      result.cov = self._calculate_covariance_matrix_(result.shgo_x)
      if result.cov is not None:
        self._calculate_uncertainties_correlations_()

    return result



  # Dual Annealing optimization ----------------------------------------------

  def dual_annealing(self, params=None, verbose=False, **method_kwgs):
    """
    Dual Annealing is probabilistic technique for approximating the global
    optimum of a given function. Specifically, it is a metaheuristic to
    approximate global optimization in a large search space for an
    optimization problem. It is often used when the search space is
    discrete (e.g., the traveling salesman problem). For problems where
    finding an approximate global optimum is more important than finding a
    precise local optimum in a fixed amount of time, simulated annealing
    may be preferable to alternatives such as gradient descent.
    -- Wikipedia

    In:
         params:  Set of parameters, must be ipanema.parameter.Parameters
    method_kwgs:  Keyword-arguments passed to the minimization algorithm.

    Out:
              0:  Optimizer result object that in general include all info
                  that the selected method provides.
    """

    result = self.prepare_fit(params=params)
    result.method = 'dual_annealing'

    da_kwgs = dict(
                    maxiter=1000,
                    local_search_options={},
                    initial_temp=5230.0,
                    restart_temp_ratio=2e-05,
                    visit=2.62,
                    accept=-5.0,
                    maxfun=10000000.0,
                    seed=None,
                    no_local_search=False,
                    callback=None,
                    x0=None
                  )
    da_kwgs.update(self.miner_kwgs)
    da_kwgs.update(method_kwgs)

    freeing = np.asarray([par.free for par in self.params.values()])
    bounds = np.asarray([(par.min, par.max) for par in
                         self.params.values()])[freeing]

    if not np.all(np.isfinite(bounds)):
      raise ValueError('dual_annealing requires finite bounds for all'
                       ' freeing parameters')

    try:
      fcn = scipy_dual_annealing(self._wrapper_scipy_minimize_, bounds,
                                 **da_kwgs)
    except KeyboardInterrupt:
      pass

    if not result.aborted:
      for attr, value in fcn.items():
        if attr in ['success', 'message']:
          setattr(result, attr, value)
        else:
          setattr(result, 'da_{}'.format(attr), value)
      result.residual = self._residual_(result.da_x, False)
      result.nfev -= 1

    result._compute_statistics_()

    # calculate the cov and estimate uncertanties/correlations
    if (not result.aborted and self.calc_covar):
      result.cov = self._calculate_covariance_matrix_(result.da_x)
      if result.cov is not None:
        self._calculate_uncertainties_correlations_()

    return result



  # Optimization launcher ----------------------------------------------------

  def optimize(self, params=None, method='lbfgsb', **method_kwgs):
    """
    Perform the minimization.

    In:
    0.123456789:
         params:  Set of parameters, must be ipanema.parameter.Parameters
         method:  Optimizer to use, there are...
                  GRADIENT-BASED:
                    - 'bfgs': Broyden–Fletcher–Goldfarb–Shanno
                    - 'lbfgsb': Limited-memory BFGS with bounds
                    - 'migrad': CERN Minuit (DFP method) calling migrad
                    - 'minuit': CERN Minuit (DFP method) callin minuit
                    - 'minos': CERN Minuit (DFP method) calling minos
                    - 'leastsq': Levenberg-Marquardt
                    - 'least_squares': Trust Region Reflective method
                    - 'powell': Powell
                    - 'cg': Conjugate-Gradient
                    - 'newton': Newton-CG
                    - 'cobyla': Constrained optimization by linear approx
                  STOCHASTIC-BASED:
                    - 'emmcc': Maximum likelihood via Monte-Carlo Markov Chain
                    - 'basinhopping': basinhopping (~Metropolis–Hastings)
                    - 'dual_annealing': Dual Annealing optimization
                    – 'multinest': -> not yet
                  GENETIC ALGORITHMS:
                    - 'deap': -> not yet
                  HEURISTIC:
                    - 'differential_evolution': differential evolution
                    - 'nelder': Nelder-Mead
                  LIPSCHIZ FUNCTIONS:
                    - 'shgo': Simplicial Homology Global Optimization
                  CLASSIFY these:
                    - 'tnc': Truncated Newton
                    - 'trust-ncg': Newton-CG trust-region
                    - 'trust-exact': nearly exact trust-region
                    - 'trust-krylov': Newton GLTR trust-region
                    - 'trust-constr': trust-region for constrained optimization
                    - 'dogleg': Dog-leg trust-region
                    - 'slsqp': Sequential Linear Squares Programming

    method_kwgs:  Keyword-arguments to be passed to the underlying
                  minimization algorithm (the selected method).
    """
    kwgs = {'params': params}
    kwgs.update(self.miner_kwgs)
    kwgs.update(method_kwgs)
    miner = method.lower()
    if   miner == 'lm':
      #function = self.least_squares
      function = self.levenberg_marquardt
      #kwgs['method'] = 'lm'
    elif miner.startswith('least'):
      function = self.least_squares
      kwgs['method'] = 'least_squares'
    elif miner == 'brute':
      function = self.brute
    elif miner == 'basinhopping':
      function = self.basinhopping
    elif miner == 'emcee':
      function = self.emcee
    elif miner == 'shgo':
      function = self.shgo
    elif miner == 'minuit':
      function = self.minuit
      kwgs['method'] = 'hesse'
    elif miner == 'minos':
      function = self.minuit
      kwgs['method'] = 'minos'
    elif miner == 'dual_annealing':
      function = self.dual_annealing
    else:
      function = self.scalar_optimize
      for k, v in SCIPY_METHODS.items():
        if (k.lower().startswith(miner) or v.lower().startswith(miner)):
          kwgs['method'] = v
    return function(**kwgs)



################################################################################



################################################################################
# Optimize function ############################################################

def optimize(fcn_call, params, method='lbfgsb',
             fcn_args=None, fcn_kwgs=None,
             model_call=None,
             scale_covar=True, policy='filter', calc_covar=True,
             residual_reduce=None,
             verbose=False, **method_kwgs):
  """
  Search for the minimum of an objective function with one of the provided
  methods.
  This function is a Optimizer wrapper only, so the same can be achieved if
      fit = Optimizer(...)
      fit.optimize(method=desired_method)
  the main reason to use this, is to always reset the Optimizer, that is often
  the best practice to avoid mistakes/errors.
  This function do not overwrite the params object that is provided, instead
  the fitted params are stored in OptimizerResult.params.

  In:
  0.123456789:
     fcn_call:  A callable function fcn_call(pars, *fcn_args, **fcn_kwgs)
                that returns an array. Optimizer will handle the sum and how
                to do it.
                callable
         pars:  Set of parametes
                ipanema.Parameters
       method:  Optimizer to use. Check Optimizer.optimize help to see all of
                them. They are also all listed in:
                    ipanema.all_optimize_methods
                string (default='lbfgsb')
     fcn_args:  Set of positional arguments that fcn needs (or handles)
                tuple
     fcn_kwgs:  Set of keyword arguments that fcn needs (or handles)
                dict
  method_kwgs:  Set of keyword arguments passed to the optimizer method. If
                the optimizer cannot handle them there will be errors.
                dict

  Out:
            0:  Optimizer result object that in general include all info that
                the selected method provides (at least the most useful one).
  """
  t0 = timer()
  fitter = Optimizer(fcn_call, params, fcn_args=fcn_args, fcn_kwgs=fcn_kwgs,
                     model_call=model_call, scale_covar=scale_covar,
                     policy=policy, residual_reduce=residual_reduce,
                     calc_covar=calc_covar, **method_kwgs)
  result = fitter.optimize(method=method)
  tf = timer()-t0
  if verbose:
    result.params.print()
    hours, rem = divmod(tf, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Optimization finished in {hours}h {minutes}m {seconds:2.3}s.')
  return result

################################################################################
