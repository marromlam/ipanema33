"""Functions to display fitting results and confidence intervals."""
from math import log10
import re
import warnings
import numpy as np




def __parse_attr(obj, attr, length=11):
  val = getattr(obj, attr, None)
  if val is None:
    return 'not avaliable'
  elif isinstance(val, int):
    return '%d' % val
  elif isinstance(val, float):
    return val
  return repr(val)




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
                  if (name != name2 and name2 in par.correl and
                          abs(par.correl[name2]) > min_correl):
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











def ci_report(ci, with_offset=True, ndigits=5):
    """Return text of a report for confidence intervals.

    Parameters
    ----------
    with_offset : bool, optional
        Whether to subtract best value from all other values (default is True).
    ndigits : int, optional
        Number of significant digits to show (default is 5).

    Returns
    -------
    str
       Text of formatted report on confidence intervals.

    """
    maxlen = max([len(i) for i in ci])
    buff = []
    add = buff.append

    def convp(x):
        """Convert probabilities into header for CI report."""
        if abs(x[0]) < 1.e-2:
            return "_BEST_"
        return "%.2f%%" % (x[0]*100)

    title_shown = False
    fmt_best = fmt_diff = "{0:.%if}" % ndigits
    if with_offset:
        fmt_diff = "{0:+.%if}" % ndigits
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen+1)] + [i.rjust(ndigits+5)
                                                for i in map(convp, row)]))
            title_shown = True
        thisrow = [" %s:" % name.ljust(maxlen)]
        offset = 0.0
        if with_offset:
            for cval, val in row:
                if abs(cval) < 1.e-2:
                    offset = val
        for cval, val in row:
            if cval < 1.e-2:
                sval = fmt_best.format(val)
            else:
                sval = fmt_diff.format(val-offset)
            thisrow.append(sval.rjust(ndigits+5))
        add("".join(thisrow))

    return '\n'.join(buff)


def report_ci(ci):
    """Print a report for confidence intervals."""
    print(ci_report(ci))
