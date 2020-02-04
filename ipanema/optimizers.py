# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                                 OPTIMIZER                                    #
#                                                                              #
#     Author: Marcos Romero                                                    #
#    Created: 04 - dec - 2019                                                  #
#                                                                              #
#                                                                              #
#                                                                              #
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
#from scipy.optimize import brute as scipy_brute
from scipy.optimize import differential_evolution, least_squares
from scipy.optimize import dual_annealing as scipy_dual_annealing
from scipy.optimize import shgo as scipy_shgo
import emcee

# Scipy functions
from scipy.stats import cauchy as cauchy_dist
from scipy.stats import norm as norm_dist
from scipy.version import version as scipy_version

import six
import uncertainties as unc

# Ipanema modules
from .parameter import Parameter, Parameters
from .utils.printfuncs import fitreport_html_table








# define the namedtuple here so pickle will work with the OptimizerResult
Candidate = namedtuple('Candidate', ['params', 'score'])


def asteval_with_uncertainties(*vals, **kwargs):
    """Calculate object value, given values for variables.

    This is used by the uncertainties package to calculate the
    uncertainty in an object even with a complicated expression.

    """
    _obj = kwargs.get('_obj', None)
    _pars = kwargs.get('_pars', None)
    _names = kwargs.get('_names', None)
    _asteval = _pars._asteval
    if (_obj is None or _pars is None or _names is None or
            _asteval is None or _obj._expr_ast is None):
        return 0
    for val, name in zip(vals, _names):
        _asteval.symtable[name] = val
    return _asteval.eval(_obj._expr_ast)


wrap_ueval = unc.wrap(asteval_with_uncertainties)


def eval_stdev(obj, uvars, _names, _pars):
    """Evaluate uncertainty and set .stdev for a parameter `obj`.

    Given the uncertain values `uvars` (a list of unc.ufloats), a
    list of parameter names that matches uvars, and a dict of param objects,
    keyed by name.

    This uses the uncertainties package wrapped function to evaluate the
    uncertainty for an arbitrary expression (in obj._expr_ast) of parameters.

    """
    if not isinstance(obj, Parameter) or getattr(obj, '_expr_ast', None) is None:
        return
    uval = wrap_ueval(*uvars, _obj=obj, _names=_names, _pars=_pars)
    try:
        obj.stdev = uval.std_dev
    except Exception:
        obj.stdev = 0


class OptimizerException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        return "{}".format(self.msg)


class AbortFitException(OptimizerException):
    """Raised when a fit is aborted by the user."""

    pass





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
  'newton':                 'Newton-CG',
  'lbfgsb':                 'L-BFGS-B',
  'tnc':                    'TNC',
  'cobyla':                 'COBYLA',
  'slsqp':                  'SLSQP',
  'dogleg':                 'dogleg',
  'trust-ncg':              'trust-ncg',
  'hesse':                  'hesse',
  'lm':                     'lm',
  'least_squares':          'least_squares'
}

STOCHASTIC_METHODS = {
  'emmcc':                  'emmcc',
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


# FIXME: update this when incresing the minimum scipy version
major, minor, micro = scipy_version.split('.', 2)
if (int(major) >= 1 and int(minor) >= 1):
    SCIPY_METHODS.update({'trust-constr': 'trust-constr'})
if int(major) >= 1:
    SCIPY_METHODS.update({'trust-exact': 'trust-exact',
                           'trust-krylov': 'trust-krylov'})







def _lnprior_(theta, bounds):
   """
   Calculate an improper uniform log-prior probability.

   Parameters
   ----------
   theta : sequence
       Float parameter values (only those being varied).
   bounds : np.ndarray
       Lower and upper bounds of parameters that are freeing.
       Has shape (nvary, 2).

   Returns
   -------
   lnprob : float
       Log prior probability.

   """
   if np.any(theta > bounds[:, 1]) or np.any(theta < bounds[:, 0]):
       return -np.inf
   return 0


def _lnpost_(theta, call_fcn, params, param_vary, bounds, fcnargs=(),
           userkws=None, float_behavior='posterior', is_weighted=True,
           nan_policy='raise'):
   """Calculate the log-posterior probability.

   See the `Optimizer.emcee` method for more details.

   Parameters
   ----------
   theta : sequence
       Float parameter values (only those being varied).
   call_fcn : callable
       User objective function.
   params : :class:`~lmfit.parameters.Parameters`
       The entire set of Parameters.
   param_vary : list
       The names of the parameters that are freeing.
   bounds : numpy.ndarray
       Lower and upper bounds of parameters. Has shape (nvary, 2).
   fcnargs : tuple, optional
       Extra positional arguments required for user objective function.
   userkws : dict, optional
       Extra keyword arguments required for user objective function.
   float_behavior : str, optional
       Specifies meaning of objective when it returns a float. One of:

       'posterior' - objective function returnins a log-posterior
                     probability
       'chi2' - objective function returns a chi2 value

   is_weighted : bool
       If `call_fcn` returns a vector of residuals then `is_weighted`
       specifies if the residuals have been weighted by data unc.
   nan_policy : str, optional
       Specifies action if `call_fcn` returns NaN values. One of:

           'raise' - a `ValueError` is raised
           'propagate' - the values returned from `call_fcn` are un-altered
           'omit' - the non-finite values are filtered


   Returns
   -------
   lnprob : float
       Log posterior probability.

   """
   # the comparison has to be done on theta and bounds. DO NOT inject theta
   # values into Parameters, then compare Parameters values to the bounds.
   # Parameters values are clipped to stay within bounds.
   if np.any(theta > bounds[:, 1]) or np.any(theta < bounds[:, 0]):
       return -np.inf

   for name, val in zip(param_vary, theta):
       params[name].value = val

   userkwargs = {}
   if userkws is not None:
       userkwargs = userkws

   # update the constraints
   params.update_constraints()

   # now calculate the log-likelihood
   out = call_fcn(params, *fcnargs, **userkwargs)
   out = _handle_nans(out, nan_policy=nan_policy, handle_inf=False)

   lnprob = np.asarray(out).ravel()

   if lnprob.size > 1:
       # objective function returns a vector of residuals
       if '__lnsigma' in params and not is_weighted:
           # marginalise over a constant data uncertainty
           __lnsigma = params['__lnsigma'].value
           c = np.log(2 * np.pi) + 2 * __lnsigma
           lnprob = -0.5 * np.sum((lnprob / np.exp(__lnsigma)) ** 2 + c)
       else:
           lnprob = -0.5 * (lnprob * lnprob).sum()
   else:
       # objective function returns a single value.
       # use float_behaviour to figure out if the value is posterior or chi2
       if float_behavior == 'posterior':
           pass
       elif float_behavior == 'chi2':
           lnprob *= -0.5
       else:
           raise ValueError("float_behaviour must be either 'posterior' or"
                            " 'chi2' " + float_behavior)

   return lnprob


def _make_random_gen_(seed):
   """Turn seed into a numpy.random.RandomState instance.

   If seed is None, return the RandomState singleton used by
   numpy.random. If seed is an int, return a new RandomState instance
   seeded with seed. If seed is already a RandomState instance, return
   it. Otherwise raise ValueError.

   """
   if seed is None or seed is np.random:
       return np.random.mtrand._rand
   if isinstance(seed, (numbers.Integral, np.integer)):
       return np.random.RandomState(seed)
   if isinstance(seed, np.random.RandomState):
       return seed
   raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                    ' instance' % seed)


def _handle_nans_(arr, nan_policy='raise', handle_inf=True):
   """Specify behaviour when an array contains numpy.nan or numpy.inf.

   Parameters
   ----------
   arr : array_like
       Input array to consider.
   nan_policy : str, optional
       One of:

       'raise' - raise a `ValueError` if `arr` contains NaN (default)
       'propagate' - propagate NaN
       'omit' - filter NaN from input array
   handle_inf : bool, optional
       As well as NaN consider +/- inf.

   Returns
   -------
   filtered_array : array_like

   Note
   ----
   This function is copied, then modified, from
   scipy/stats/stats.py/_contains_nan

   """
   if nan_policy not in ('propagate', 'omit', 'raise'):
       raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")

   if handle_inf:
       handler_func = lambda x: ~np.isfinite(x)
   else:
       handler_func = isnull

   if nan_policy == 'omit':
       # mask locates any values to remove
       mask = ~handler_func(arr)
       if not np.all(mask):  # there are some NaNs/infs/missing values
           return arr[mask]

   if nan_policy == 'raise':
       try:
           # Calling np.sum to avoid creating a huge array into memory
           # e.g. np.isnan(a).any()
           with np.errstate(invalid='ignore'):
               contains_nan = handler_func(np.sum(arr))
       except TypeError:
           # If the check cannot be properly performed we fallback to omiting
           # nan values and raising a warning. This can happen when attempting to
           # sum things that are not numbers (e.g. as in the function `mode`).
           contains_nan = False
           warnings.warn("The input array could not be checked for NaNs. "
                         "NaNs will be ignored.", RuntimeWarning)

       if contains_nan:
           msg = ('NaN values detected in your input data or the output of '
                  'your objective/model function - fitting algorithms cannot '
                  'handle this! Please read https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do '
                  'for more information.')
           raise ValueError(msg)
   return arr

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
       covar :  Covariance matrix from minimization.
  param_init :  List of initial values for variable parameters using.
  init_values:  Dictionary of initial values for variable parameters.
         nfev:  Number of function evaluations.
      success:  Termination status of the optimizer if it's valid or not
    errorbars:  True if uncertainties were estimated, otherwise False.
      message:  Message about fit success.
         ier :  Integer error value from :scipydoc:`optimize.leastsq` (`leastsq` only).
lmdif_message:  Message from :scipydoc:`optimize.leastsq` (`leastsq` only).
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

  @property
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
      self.chi2 = (self.residual**2).sum()
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

  def _repr_html_(self, corr=True, min_corr=0.05):
    """
    Returns a HTML representation of parameters data.
    """
    return fitreport_html_table(self, show_correl=corr, min_correl=min_corr)

################################################################################



################################################################################
# Optimizer Object #############################################################

class Optimizer(object):
    """
    A general optimizer for curve fitting and optimization.
    """

    _err_nonparam = ("params must be a optimizer.Parameters() instance or list "
                     "of Parameters()")
    _err_maxfev = ("Too many function calls (max set to %i)!  Use:"
                   " optimize(func, params, ..., maxfev=NNN)"
                   "or set leastsq_kws['maxfev']  to increase this maximum.")

    def __init__(self, call_fcn, params,
                 fcn_args=None, fcn_kwgs=None,
                 iter_cb=None, scale_covar=True, nan_policy='raise',
                 reduce_fcn=None, calc_covar=True,
                 **method_kwgs):
      """
      Initialize the Optimizer class.

      The objective function should return the array of residuals to be
      optimized that afterwards will be reduced to a FCN. A call_fcn function
      usually needs data, uncertainties, weights..., these can be handled under
      fcn_args and fcn_kwgs. Parameters should be passed independly through
      the params argument.

      In:
         call_fcn:  Objective function that returns the residual
                    (array, same lengh as data). This function must have the
                    signature:
                        call_fcn(params, *fcn_args, **fcn_kwgs)
                    callable


           params:  Set of paramters.
                    ipanema.parameter.Parameters
         fcn_args:  Positional arguments to pass to call_fcn.
                    tuple, optional (default=None)
         fcn_kwgs:  Keyword arguments to pass to call_fcn.
                    dict, optional (default=None)
          iter_cb:  Function to be called at each fit iteration. This function
                    should have the signature:
                        iter_cb(params, iter, resid, *fcn_args, **fcn_kwgs)
                    callable, optional (default=None)
      scale_covar:  Scale covariance matrix
                    bool, optional (default=True)
       nan_policy:  When a NaN value if returned ipanema can handle it in two
                    ways: 'raise', a `ValueError` is raised or 'filter', the
                    NaN value is replaced by 1e12.
                    str, optional (default='raise')
       reduce_fcn:  Function to convert a residual array to a scalar value,
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
      self.call_fcn = call_fcn
      self.fcnargs = fcn_args
      if self.fcnargs is None: self.fcnargs = []
      self.fcnkwgs = fcn_kwgs
      if self.fcnkwgs is None: self.fcnkwgs = {}
      self.miner_kwgs = method_kwgs
      self.iter_cb = iter_cb
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
      self.reduce_fcn = self._residual_sum_
      if reduce_fcn: self.reduce_fcn = reduce_fcn
      self.params = params
      self.jacfcn = None
      self.nan_policy = nan_policy

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
      out = (array).sum()
      if math.isnan(out): out = 1e12 # work in policy
      return out


    def _residual_squared_sum_(self,array):
      """
      Reduce residual array to scalar with the squared sum.
      """
      out = (array*array).sum()
      if math.isnan(out): out = 1e12 # work in policy
      return out



    # Wrappers around call_fcn -------------------------------------------------

    def _residual_(self, fvars, apply_bounds_transformation=True):
        """Residual function used for least-squares fit.

        With the new, candidate values of `fvars` (the fitting variables),
        this evaluates all parameters, including setting bounds and
        evaluating constraints, and then passes those to the user-supplied
        function to calculate the residual.

        Parameters
        ----------
        fvars : numpy.ndarray
            Array of new parameter values suggested by the optimizer.
        apply_bounds_transformation : bool, optional
            Whether to apply lmfits parameter transformation to constrain
            parameters (default is True). This is needed for solvers without
            inbuilt support for bounds.

        Returns
        -------
        residual : numpy.ndarray
             The evaluated function values for given `fvars`.

        """
        params = self.result.params

        if fvars.shape == ():
            fvars = fvars.reshape((1,))

        if apply_bounds_transformation:
            for name, val in zip(self.result.param_vary, fvars):
                params[name].value = params[name].from_internal(val)
        else:
            for name, val in zip(self.result.param_vary, fvars):
                params[name].value = val
        params.update_constraints()

        self.result.nfev += 1

        out = self.call_fcn(params, *self.fcnargs, **self.fcnkwgs)

        if callable(self.iter_cb):
            abort = self.iter_cb(params, self.result.nfev, out,
                                 *self.fcnargs, **self.fcnkwgs)
            self._abort = self._abort or abort

        if self._abort:
            self.result.residual = out
            self.result.aborted = True
            self.result.message = "Fit aborted by user callback. Could not estimate error-bars."
            self.result.success = False
            raise AbortFitException("fit aborted by user.")
        else:
            return _handle_nans(np.asarray(out).ravel(),
                               nan_policy=self.nan_policy)



    def _minuit_wrapper_(self, *fvars, reduce = True):
      """
      Residual function used for minuit methods.

      In:
            fvars:  Array of values of parameters
                    array
           reduce:  Boolean to return FCN or vector of residuals

      Out:
                0:  Function-Call Number / vector of residuals
      """
      # Get parameters and set new proposals
      pars = self.result.params
      for name, val in zip(self.result.param_vary, fvars):
        pars[name].value = val
      pars.update_constraints()

      # Compute fcn
      fcn = self.call_fcn(pars, *self.fcnargs, **self.fcnkwgs)
      self.result.nfev += 1 # update n of fcn evals

      if callable(self.iter_cb):
        abort = self.iter_cb(pars, self.result.nfev, fcn, *self.fcnargs, **self.fcnkwgs)
        self._abort = self._abort or abort

      if self._abort:
        self.result.residual = fcn
        self.result.aborted  = True
        self.result.message  = "Optimization aborted by user callback."
        self.result.message += "Could not give a complete result."
        self.result.success  = False
        raise AbortFitException("Optimization aborted by user.")
      elif reduce:
        return self.reduce_fcn(fcn)
      else:
        return fcn



    def _wrapper_scipy_minimize_(self, fvars, reduce = True):
      """
      Wrapper function for scipy.minimize methods.

      In:
            fvars:  Array of values of parameters
                    array
           reduce:  Boolean to return FCN or vector of residuals

      Out:
                0:  Function-Call Number / vector of residuals

      """
      if self.result.method in ['shgo', 'dual_annealing']:
        apply_bounds_transformation = False
      else:
        apply_bounds_transformation = True
      fcn = self._residual_(fvars, apply_bounds_transformation)
      if reduce:
        return self.reduce_fcn(fcn)
      else:
        return fcn



    # Statistics calculators ---------------------------------------------------

    def _calculate_covariance_matrix_(self, fvars):
      """
      The covariance matrix.

      In:
            fvars:  Array of optimal & free values of parameters
                    array

      Out:
                0:  Covariance matrix
                    array
      """
      warnings.filterwarnings(action="ignore", module="scipy",
                              message="^internal gelsd")

      nfev = deepcopy(self.result.nfev)
      try:
        hessian = ndt.Hessian(self._wrapper_scipy_minimize_)(fvars)
        cov = 2 * inv(hessian)
      except (LinAlgError, ValueError):
        return None
      finally:
        self.result.nfev = nfev

      return cov



    def _int2ext_cov_(self, cov_int, fvars):
      """
      Transform covariance matrix to external parameter space.
      According to Minuit,
          cov_ext = np.dot(grad.T, grad) * cov_int

      In:
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
      cov_ext = cov_int * np.dot(g.T, g)                           # minuit-like
      return cov_ext



    def _calculate_uncertainties_correlations_(self):
      """
      Calculate parameter uncertainties and correlations.
      """
      self.result.errorbars = True

      if self.scale_covar:
        self.result.covar *= self.result.chi2red

      vbest = np.atleast_1d([self.result.params[name].value for name in
                             self.result.param_vary])

      has_expr = False
      for par in self.result.params.values():
          par.stdev, par.correl = 0, None
          has_expr = has_expr or par.expr is not None

      for ivar, name in enumerate(self.result.param_vary):
          par = self.result.params[name]
          par.stdev = sqrt(self.result.covar[ivar, ivar])
          par.correl = {}
          try:
              self.result.errorbars = self.result.errorbars and (par.stdev > 0.0)
              for jvar, varn2 in enumerate(self.result.param_vary):
                  if jvar != ivar:
                      par.correl[varn2] = (self.result.covar[ivar, jvar] /
                                           (par.stdev * sqrt(self.result.covar[jvar, jvar])))
          except ZeroDivisionError:
              self.result.errorbars = False

      if has_expr:
          try:
              uvars = unc.correlated_values(vbest, self.result.covar)
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



    def _configure_minuit_(self, pars, **kwargs):
      """
      Configure minuit things
      """
      config = {}
      for par in pars.keys():
        if par in self.result.param_vary:
          config.update(self._parameter_minuit_config_(pars[par]))
      config.update(kwargs)
      return config



    def _parameter_minuit_config_(self,par):
      """
      Only free params go to Minuit
      """
      out = {par.name: par.init_value}
      lims = [None,None]
      if abs(par.min) != np.inf: lims[0] = par.min
      if abs(par.max) != np.inf: lims[1] = par.max
      out.update ({"limit_" + par.name: tuple(lims)})
      return out



    # Prepara and unprepare fit functions --------------------------------------

    def prepare_fit(self, params=None):
      """
      In:
           params:  Parameters to use.
                    ipanema.parameter.Parameters, optional
      Out:
                0:  OptimizerResult object prepared to perform fits
                    optimizers.OptimizerResult
      """
      # Construct a OptimizerResult to store fitting-info
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
            raise OptimizerException(self._err_nonparam)
          else:
            result.params[par.name] = par
      elif self.params is None:
          raise OptimizerException(self._err_nonparam)

      # Check paraeter atributes and consistency
      result.param_vary = []
      result.param_init = []
      result.params.update_constraints()
      result.nfev = 0
      result.errorbars = False
      result.aborted = False
      for name, par in self.result.params.items():
        par.stdev = None
        par.correl = None
        # Which parameters are defined by expressions?
        if par.expr is not None:
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
      result.init_values = {n: v for n, v in zip(result.param_vary,
                                                 result.param_init)}

      # Set up reduce function
      if not callable(self.reduce_fcn):
        self.reduce_fcn = self._residual_sum_
      return result



    def unprepare_fit(self):
      """
      Clean fit state, so that subsequent fits need to call prepare_fit().

      removes AST compilations of constraint expressions.

      """
      pass



    # Minuit method ------------------------------------------------------------

    def minuit(self, params=None, method='hesse', **method_kwgs):
      """
      Optimization using Minuit.
      """
      result = self.prepare_fit(params=params)
      result.method = method

      minuit_kws = dict(errordef=1,
                        print_level=-1,
                        pedantic=False)
      minuit_kws.update(method_kwgs)

      try:
        ret = minuit(self._minuit_wrapper_,
                     forced_parameters=self.result.param_vary,
                     **self._configure_minuit_(result.params, **minuit_kws)
                    )
        ret.set_strategy(1);
        if method == 'migrad':
          ret.migrad(ncall=1000 * (len(result.param_init)+1))
          ret.migrad()
        elif method == 'hesse':
          ret.migrad(ncall=1000 * (len(result.param_init)+1))
          ret.migrad(); ret.hesse()
        elif method == 'minos':
          ret.migrad(ncall=1000 * (len(result.param_init)+1))
          ret.migrad(); ret.hesse(); ret.minos()
      except AbortFitException:
        pass

      if not result.aborted:
        # return minuit class (you can keep optimizing, but without ipanema)
        result._minuit = ret
        # calculate fit statistics
        result.x = np.atleast_1d(ret.args)
        result.residual = self._minuit_wrapper_(*result.x, reduce = False)
        result.nfev -= 1
        result._compute_statistics_()
        # calculate the cov and estimate uncertanties/correlations
        result.cov = np.matrix(ret.matrix())
        result.invcov = np.matrix(np.linalg.inv(result.cov))
        for par in self.result.param_vary:
          result.params[par].value = ret.values[par]
          result.params[par].stdev = ret.errors[par]

      return result



    # Scipy.optimize methods handler -------------------------------------------

    def scalar_optimize(self, params=None, method='BFGS', **method_kwgs):
      """
      Optimization using scipy.optimize functions. For info about the Methods
      please check :scipydoc:`optimize.optimize`
      """
      result = self.prepare_fit(params=params)
      result.method = method
      variables = result.param_init
      params = result.params

      scpmin_kws = dict(
                          method=method,
                          options={'maxiter': 1000 * (len(variables) + 1)}
                        )
      scpmin_kws.update(self.kws)
      scpmin_kws.update(method_kwgs)

      # hess supported only in some methods
      if 'hess' in scpmin_kws and method not in ('Newton-CG', 'dogleg',
                                               'trust-constr', 'trust-ncg',
                                               'trust-krylov', 'trust-exact'):
          scpmin_kws.pop('hess')

      # jac supported only in some methods (and Dfun could be used...)
      if 'jac' not in scpmin_kws and scpmin_kws.get('Dfun', None) is not None:
          self.jacfcn = scpmin_kws.pop('jac')
          scpmin_kws['jac'] = self.__jacobian

      if 'jac' in scpmin_kws and method not in ('CG', 'BFGS', 'Newton-CG',
                                              'L-BFGS-B', 'TNC', 'SLSQP',
                                              'dogleg', 'trust-ncg',
                                              'trust-krylov', 'trust-exact'):
          self.jacfcn = None
          scpmin_kws.pop('jac')

      # workers / updating keywords only supported in differential_evolution
      for kwd in ('workers', 'updating'):
          if kwd in scpmin_kws and method != 'differential_evolution':
              scpmin_kws.pop(kwd)

      if method == 'differential_evolution':
          for par in params.values():
              if (par.free and
                      not (np.isfinite(par.min) and np.isfinite(par.max))):
                  raise ValueError('differential_evolution requires finite '
                                   'bound for all freeing parameters')

          _bounds = [(-np.pi / 2., np.pi / 2.)] * len(variables)
          kwargs = dict(args=(), strategy='best1bin', maxiter=None,
                        popsize=15, tol=0.01, mutation=(0.5, 1),
                        recombination=0.7, seed=None, callback=None,
                        disp=False, polish=True, init='latinhypercube',
                        atol=0, updating='immediate', workers=1)

          for k, v in scpmin_kws.items():
              if k in kwargs:
                  kwargs[k] = v

          # keywords 'updating' and 'workers' are introduced in SciPy v1.2
          # FIXME: remove after updating the requirement >= 1.2
          if int(major) == 0 or (int(major) == 1 and int(minor) < 2):
              kwargs.pop('updating')
              kwargs.pop('workers')

          try:
              ret = differential_evolution(self._wrapper_scipy_minimize_, _bounds, **kwargs)
          except AbortFitException:
              pass

      else:
          try:
              ret = scipy_minimize(self._wrapper_scipy_minimize_, variables, **scpmin_kws)
          except AbortFitException:
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
          result.residual = self._residual_(result.x)
          result.nfev -= 1

      result._compute_statistics_()

      # calculate the cov and estimate uncertanties/correlations
      if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
              len(result.residual) > len(result.param_vary)):
          _covar_ndt = self._calculate_covariance_matrix(result.x)
          if _covar_ndt is not None:
              result.covar = self._int2ext_cov(_covar_ndt, result.x)
              self._calculate_uncertainties_correlations_()

      return result



    # MCMC Hammer --------------------------------------------------------------

    def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
              ntemps=1, pos=None, reuse_sampler=False, workers=1,
              float_behavior='posterior', is_weighted=True, seed=None,
              progress=True):
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
                    to be if your `call_fcn` returns an array and `is_weighted is
                    False`.
                    array, optional (default=None)
    reuse_sampler:  If emcee was already used to optimize a function and there
                    is no change in the parameters, then one can continue
                    drawing from the same sampler. This argument skips emcee
                    to load other arguments, so be aware.
                    bool, optional (default=False)
          workers:  For parallelization of sampling.
                    pool-like or int, optional (default=1)
   float_behavior:  Whether the function-call method returns a log-posterior
                    probability ('posterior') or a chi2 ('chi2')
                    str, optional (default='posterior').
      is_weighted:  If True, emcee will supose that residuals have been
                    divided by the true measurement uncertainty; if False,
                    is assumed that unweighted residuals are passed.
                    In this second case `emcee` will employ a positive
                    measurement uncertainty during the sampling. This
                    measurement uncertainty will be present in the output
                    params and output chain with the name `__lnsigma`.
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

      # check whether the call_fcn returns a vector of residuals
      out = self.call_fcn(params, *self.fcnargs, **self.fcnkwgs)
      out = np.asarray(out).ravel()
      if out.size > 1 and is_weighted is False:
          # we need to marginalise over a constant data uncertainty
          if '__lnsigma' not in params:
              # __lnsigma should already be in params if is_weighted was
              # previously set to True.
              params.add('__lnsigma', value=0.01, min=-np.inf, max=np.inf, free=True)
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
          if param.expr is not None:
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
      sampler_kwargs = {}
      if isinstance(workers, int) and workers > 1:
          auto_pool = multiprocessing.Pool(workers)
          sampler_kwargs['pool'] = auto_pool
      elif hasattr(workers, 'map'):
          sampler_kwargs['pool'] = workers

      # function arguments for the log-probability functions
      # these values are sent to the log-probability functions by the sampler.
      lnprob_args = (self.call_fcn, params, result.param_vary, bounds)
      lnprob_kwargs = {'is_weighted': is_weighted,
                       'float_behavior': float_behavior,
                       'fcnargs': self.fcnargs,
                       'userkws': self.fcnkwgs,
                       'nan_policy': self.nan_policy}

      if ntemps > 1:
          # the prior and likelihood function args and kwargs are the same
          sampler_kwargs['loglargs'] = lnprob_args
          sampler_kwargs['loglkwargs'] = lnprob_kwargs
          sampler_kwargs['logpargs'] = (bounds,)
      else:
          sampler_kwargs['args'] = lnprob_args
          sampler_kwargs['kwargs'] = lnprob_kwargs

      # set up the random number generator
      rng = _make_random_gen(seed)

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
                                         _lnpost, _lnprior, **sampler_kwargs)
      else:
          p0 = 1 + rng.randn(nwalkers, self.nvary) * 1.e-4
          p0 *= var_arr
          self.sampler = emcee.EnsembleSampler(nwalkers, self.nvary,
                                               _lnpost, **sampler_kwargs)

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
      out = self.call_fcn(params, *self.fcnargs, **self.fcnkwgs)
      result.residual = _handle_nans(out, nan_policy=self.nan_policy, handle_inf=False)

      # If uncertainty was automatically estimated, weight the residual properly
      if (not is_weighted) and (result.residual.size > 1):
          if '__lnsigma' in params:
              result.residual = result.residual/np.exp(params['__lnsigma'].value)

      # Calculate statistics for the two standard cases:
      if isinstance(result.residual, ndarray) or (float_behavior == 'chi2'):
        result._compute_statistics_()

      # Handle special case unique to emcee:
      # This should eventually be moved into result._compute_statistics_.
      elif float_behavior == 'posterior':
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



    # Standard least-squares method --------------------------------------------

    def least_squares(self, params=None, **kws):
        """Least-squares minimization using :scipydoc:`optimize.least_squares`.

        This method wraps :scipydoc:`optimize.least_squares`, which has inbuilt
        support for bounds and robust loss functions. By default it uses the
        Trust Region Reflective algorithm with a linear loss function (i.e.,
        the standard least-squares problem).

        Parameters
        ----------
        params : :class:`~lmfit.parameter.Parameters`, optional
           Parameters to use as starting point.
        **kws : dict, optional
            Optimizer options to pass to :scipydoc:`optimize.least_squares`.

        Returns
        -------
        :class:`OptimizerResult`
            Object containing the optimized parameter and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`OptimizerResult`.

        """
        result = self.prepare_fit(params)
        result.method = 'least_squares'

        replace_none = lambda x, sign: sign*np.inf if x is None else x

        start_vals, lower_bounds, upper_bounds = [], [], []
        for vname in result.param_vary:
            par = self.params[vname]
            start_vals.append(par.value)
            lower_bounds.append(replace_none(par.min, -1))
            upper_bounds.append(replace_none(par.max, 1))

        try:
            ret = least_squares(self._residual_, start_vals,
                                bounds=(lower_bounds, upper_bounds),
                                kwargs=dict(apply_bounds_transformation=False),
                                **kws)
            result.residual = ret.fun
        except AbortFitException:
            pass

        # note: upstream least_squares is actually returning
        # "last evaluation", not "best result", but we do this
        # here for consistency, and assuming it will be fixed.
        if not result.aborted:
            result.residual = self._residual_(ret.x, False)
            result.nfev -= 1
        result._compute_statistics_()

        if not result.aborted:
            for attr in ret:
                setattr(result, attr, ret[attr])

            result.x = np.atleast_1d(result.x)

            # calculate the cov and estimate uncertainties/correlations
            try:
                hess = np.matmul(ret.jac.T, ret.jac)
                result.covar = np.linalg.inv(hess)
                self._calculate_uncertainties_correlations()
            except LinAlgError:
                pass

        return result



    # Levenberg-Marquardt Optimization method ----------------------------------

    def levenberg_marquardt(self, params=None, **kws):
        """
        Use Levenberg-Marquardt minimization to perform a fit.

        It assumes that the input Parameters have been initialized, and
        a function to optimize has been properly set up.
        When possible, this calculates the estimated uncertainties and
        variable correlations from the covariance matrix.

        This method calls :scipydoc:`optimize.leastsq`.
        By default, numerical derivatives are used, and the following
        arguments are set:

        +------------------+----------------+------------------------------------------------------------+
        | :meth:`leastsq`  |  Default Value | Description                                                |
        | arg              |                |                                                            |
        +==================+================+============================================================+
        |   xtol           |  1.e-7         | Relative error in the approximate solution                 |
        +------------------+----------------+------------------------------------------------------------+
        |   ftol           |  1.e-7         | Relative error in the desired sum of squares               |
        +------------------+----------------+------------------------------------------------------------+
        |   maxfev         | 2000*(nvar+1)  | Maximum number of function calls (nvar= # of variables)    |
        +------------------+----------------+------------------------------------------------------------+
        |   Dfun           | None           | Function to call for Jacobian calculation                  |
        +------------------+----------------+------------------------------------------------------------+

        Parameters
        ----------
        In:
        0.123456789:
             params:  Set of parameters.
                      ipanema.parameter.Parameters, optional
              **kws:  Keyword-arguments passed to the minimization algorithm.
                      dict, optional

        Out:
                  0:  Optimizer result object that in general include all info
                      that the selected method provides.

        """
        result = self.prepare_fit(params=params)
        result.method = 'Levenberg-Marquardt (lm)'
        result.nfev -= 2  # correct for "pre-fit" initialization/checks
        variables = result.param_init
        nvars = len(variables)
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7, col_deriv=False,
                     gtol=1.e-7, maxfev=2000*(nvars+1), Dfun=None)

        lskws.update(self.kws)
        lskws.update(kws)

        self.col_deriv = False
        if lskws['Dfun'] is not None:
            self.jacfcn = lskws['Dfun']
            self.col_deriv = lskws['col_deriv']
            lskws['Dfun'] = self.__jacobian

        # suppress runtime warnings during fit and error analysis
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')

        try:
          lsout = scipy_leastsq(self._residual_, variables, **lskws)
        except AbortFitException:
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
            result.covar = self._int2ext_cov(_cov, _best)
            # calculate parameter uncertainties and correlations
            self._calculate_uncertainties_correlations()
        else:
            result.message = '%s Could not estimate error-bars.' % result.message

        np.seterr(**orig_warn_settings)

        return result



    # Basin - Hopping method ---------------------------------------------------
    def basinhopping(self, params=None, **method_kwgs):
      """
      shit shit shit
      """
      result = self.prepare_fit(params=params)
      result.method = 'basinhopping'

      basinhopping_kwgs = dict(
                                niter=100,
                                T=1.0,
                                stepsize=0.5,
                                optimizer_kwargs={},
                                take_step=None,
                                accept_test=None,
                                callback=None,
                                interval=50,
                                disp=False,
                                niter_success=None,
                                seed=None
                              )

      basinhopping_kwgs.update(self.kws)
      basinhopping_kwgs.update(method_kwgs)

      x0 = result.param_init

      try:
        ret = scipy_basinhopping(self._wrapper_scipy_minimize_, x0,
                                 **basinhopping_kwgs)
      except AbortFitException:
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
          result.covar = self._int2ext_cov_(_covar_ndt, ret.x)
          self._calculate_uncertainties_correlations_()

      return result



    # Simplicial Homology Global Optimization method ---------------------------

    def shgo(self, params=None, **method_kwgs):
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
                        optimizer_kwargs=None,
                        options=None,
                        sampling_method='simplicial'
                      )

      shgo_kwgs.update(self.kws)
      shgo_kwgs.update(method_kwgs)

      freeing = np.asarray([par.free for par in self.params.values()])
      bounds = np.asarray([(par.min, par.max) for par in
                           self.params.values()])[freeing]

      try:
        ret = scipy_shgo(self._wrapper_scipy_minimize_, bounds, **shgo_kwgs)
      except AbortFitException:
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
        result.covar = self._calculate_covariance_matrix(result.shgo_x)
        if result.covar is not None:
          self._calculate_uncertainties_correlations()

      return result



    # Dual Annealing optimization ----------------------------------------------

    def dual_annealing(self, params=None, **method_kwgs):
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
      da_kwgs.update(self.kws)
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
      except AbortFitException:
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
        result.covar = self._calculate_covariance_matrix_(result.da_x)
        if result.covar is not None:
          self._calculate_uncertainties_correlations_()

      return result



    # Optimization launcher ----------------------------------------------------

    def optimize(self, params=None, method='bfgs', **method_kwgs):
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
                      - 'hesse': CERN Minuit (DFP method) callin hesse
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
      kwargs = {'params': params}
      kwargs.update(self.miner_kwgs)
      kwargs.update(method_kwgs)

      user_method = method.lower()
      if user_method == 'lm':
        function = self.levenberg_marquardt
      elif user_method.startswith('least'):
        function = self.least_squares
      elif user_method == 'brute':
        function = self.brute
      elif user_method == 'basinhopping':
        function = self.basinhopping
      elif user_method == 'emcee':
        function = self.emcee
      elif user_method == 'shgo':
        function = self.shgo
      elif user_method[:6] == 'migrad':
        function = self.minuit
        kwargs['method'] = 'migrad'
      elif user_method[:6] == 'hesse':
        function = self.minuit
        kwargs['method'] = 'hesse'
      elif user_method[:6] == 'minos':
        function = self.minuit
        kwargs['method'] = 'minos'
      elif user_method == 'dual_annealing':
        function = self.dual_annealing
      else:
        function = self.scalar_optimize
        for key, val in SCIPY_METHODS.items():
          if (key.lower().startswith(user_method) or
              val.lower().startswith(user_method)):
            kwargs['method'] = val
      return function(**kwargs)



################################################################################



################################################################################
# Optimize function ############################################################

def optimize(fcn, params, method='bfgs', args=None, kwgs=None,
             iter_cb=None, scale_covar=True, nan_policy='raise',
             reduce_fcn=None, calc_covar=True,
             verbose=False,
             **method_kwgs):
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
          fcn:  A callable function fcn(pars, *fcnargs, **fcnkwgs) that returns
                an array. Optimizer will handle the sum and how to do it.
         pars:  Set of parametes, should be a Parameters object
       method:  Optimizer to use. Check Optimizer.optimize help to see all of
                them.
      fcnargs:  Set of positional arguments that fcn needs (or handles)
      fcnkwgs:  Set of keyword arguments that fcn needs (or handles)
     minerkws:  Set of keyword arguments passed to the optimizer method. If
                the optimizer cannot handle them there will be errors.

  Out:
            0:  Optimizer result object that in general include all info that
                the selected method provides (at least the most useful one).
  """
  #print("\n\n" + 80*"=" + "\n= Optimizing\n" + 80*"=")
  t0 = timer()
  fitter = Optimizer(fcn, params, fcn_args=args, fcn_kwgs=kwgs,
                     iter_cb=iter_cb, scale_covar=scale_covar,
                     nan_policy=nan_policy, reduce_fcn=reduce_fcn,
                     calc_covar=calc_covar, **method_kwgs)
  result = fitter.optimize(method=method)
  tf = timer()-t0
  if verbose:
    result.params.print()
    print('Optimization finished in %.4f minutes.' % (tf/60) )
  return result

################################################################################
