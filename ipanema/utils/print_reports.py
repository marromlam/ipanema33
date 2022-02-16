################################################################################
#                                                                              #
#                                   REPORTs                                    #
#                                                                              #
################################################################################

import numpy as np

__all__ = ['fit_report']

################################################################################
################################################################################

def __parse_attr(obj, attr, length=11):
  val = getattr(obj, attr, None)
  if val is None:
    return 'not avaliable'
  elif isinstance(val, int):
    return '%d' % val
  elif isinstance(val, float):
    return val
  elif isinstance(val, str):
    return val
  return repr(val)

################################################################################



################################################################################
################################################################################

def fit_report(result, show_correl=True, min_correl=0.05, as_string=False):
  """
  Generate a report of the fitting results.

  In:
  0.123456789:
       result:  Fit result.
                ipanema.OptimizerResult
  show_correl:  Whether to show correlations.
                bool, optional (default=True)
  show_correl:  Cut-off to printout correlations. Correlations smaller than
                this number will not be printed.
                float, optional (default=0.05)
    as_string:  Whether to print the report or to give it as string to be
                dumped after.
                bool, optional (default=False)
  Out:
            0:  Fit report if as_string=False, else void.
                string

  """
  print_out = []; add = print_out.append

  if result is not None:
    add(f"\nFit Statistics")
    add(f"{80*'-'}")
    add(f"{'method:':>30} {__parse_attr(result, 'method')}")
    add(f"{'# fcn calls:':>30} {__parse_attr(result, 'nfev')}")
    add(f"{'# data points:':>30} {__parse_attr(result, 'ndata')}")
    add(f"{'# degrees of freedom:':>30} {__parse_attr(result, 'nfree')}")
    add(f"{'chi2:':>30} {__parse_attr(result, 'chi2')}")
    add(f"{'chi2/dof:':>30} {__parse_attr(result, 'chi2red')}")
    add(f"{'-2 logLikelihood:':>30} {__parse_attr(result, 'nll2')}")
    add(f"{'residual abinito:':>30} {__parse_attr(result, 'init_residual')}")
    add(f"{'Akaike info criterion:':>30} {__parse_attr(result, 'aic')}")
    add(f"{'Bayesian info criterion:':>30} {__parse_attr(result, 'bic')}")

    add(f"{'Fit messages:':>30} "+__parse_attr(result, 'message'))

    pars_free = [p for p in result.params if result.params[p].free]
    for name in pars_free:
      par = result.params[name]
      if par.init_value and np.allclose(par.value, par.init_value):
        add(f"{' ':>31}{name}: at initial value")
      if np.allclose(par.value, par.min) or np.allclose(par.value, par.max):
        add(f"{' ':>31}{name}: at boundary")

  add(f"\nParameters")
  add(f"{80*'-'}")
  add(result.params.print(
          cols=['value', 'stdev', 'reldev', 'min', 'max', 'free'],
          as_string=True)
  )

  if show_correl:
    add(f"\nCorrelations (ones lower than {min_correl} are not reported)")
    add(f"{80*'-'}")
    correls = {}; parnames = list(result.params.keys())
    for i, name in enumerate(parnames):
      par = result.params[name]
      if not par.free:
        continue
      if hasattr(par, 'correl') and par.correl is not None:
        for name2 in parnames[i+1:]:
          if (name != name2 and name2 in par.correl and abs(par.correl[name2]) > min_correl):
            correls["%s, %s" % (name, name2)] = par.correl[name2]

    sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
    sort_correl.reverse()
    if len(sort_correl) > 0:
      maxlen = max([len(k) for k in list(correls.keys())])
    for name, val in sort_correl:
      lspace = max(0, maxlen - len(name))
      add('    C(%s)%s = % .3f' % (name, (' '*30)[:lspace], val))
  if as_string:
    return '\n'.join(print_out)
  print('\n'.join(print_out))

################################################################################
