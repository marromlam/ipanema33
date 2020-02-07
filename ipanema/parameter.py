# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                           PARAMETER & PARAMETERS                             #
#                                                                              #
#     Author: Marcos Romero                                                    #
#    Created: 04 - dec - 2019                                                  #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################



from collections import OrderedDict
from copy import deepcopy
import hjson
import importlib

from asteval import Interpreter, get_ast_names, valid_symbol_name
from numpy import arcsin, array, cos, inf, isclose, nan, sin, sqrt
from numpy import inf as Infinite
import scipy.special
import uncertainties
import uncertainties as unc

#from .jsonutils import decode4js, encode4js
from .utils.printfuncs import params_html_table

SCIPY_FUNCTIONS = {'gamfcn': scipy.special.gamma}
for name in ('erf', 'erfc', 'wofz'):
    SCIPY_FUNCTIONS[name] = getattr(scipy.special, name)


def check_ast_errors(expr_eval):
  """
  Check for errors derived from asteval.
  """
  if len(expr_eval.error) > 0:
    expr_eval.raise_exception(None)






################################################################################
# Parameters ###################################################################

class Parameters(OrderedDict):
  """
  An ordered dictionary of all the Parameter objects. Note:
  * Parameter().name must be a valid Python symbol name
  * Parameters() is made only of Parameter() items
  """

  def __init__(self, asteval=None, usersyms=None, *args, **kwds):
    """
    ---
    """
    super(Parameters, self).__init__(self)

    self._asteval = asteval
    if self._asteval is None:
        self._asteval = Interpreter()

    _syms = {}
    _syms.update(SCIPY_FUNCTIONS)
    if usersyms is not None:
        _syms.update(usersyms)
    for key, val in _syms.items():
        self._asteval.symtable[key] = val

    self.update(*args, **kwds)

  def copy(self, params_in):
    """
    Parameters.copy() should always be a deepcopy.
    """
    return self.__deepcopy__(params_in)

  @classmethod
  def __copy__(cls, params_in):                            # take a look at this
    """
    Alias of __deepcopy__
    """
    c = cls()
    c.loads(hjson.loads(params_in.dumps()))
    return c

  def __deepcopy__(self, params_in):                       # take a look at this
    """
    Deep copy of params. current implementation is bullshit.
    """
    self.loads(hjson.loads(params_in.dumps()))
    return self

  def __setitem__(self, key, par):
    """
    Set a parameter by key
    """
    if key not in self:
      if not valid_symbol_name(key):
        raise KeyError("'%s' has not a valid Parameter name" % key)
    if par is not None and not isinstance(par, Parameter):
      raise ValueError("'%s' is not a Parameter" % par)
    OrderedDict.__setitem__(self, key, par)
    par.name = key
    par._expr_eval = self._asteval
    self._asteval.symtable[key] = par.value

  def __add__(self, friend):
    """
    Add Parameters objects.
    """
    if not isinstance(friend, Parameters):
      raise ValueError("'%s' is not a Parameters object" % friend)
    out = self.__copy__(self)
    pars_original = list(out.keys())
    pars_friend = list(friend.keys())
    for par in pars_friend:
      if par not in pars_original:
        out.add(friend[par])
    return out

  def __iadd__(self, friend):
    """Add/assign Parameters objects."""
    if not isinstance(friend, Parameters):
        raise ValueError("'%s' is not a Parameters object" % friend)
    params = friend.values()
    self.add_many(*params)
    return self

  def __array__(self):
    """
    Convert Parameters to array.
    """
    return array([float(k) for k in self.values()])

  def __setstate__(self, params):
    """
    Alias of add_many.
    """
    self.add_many(params)

  def eval(self, expr):
    """
    Evaluate a statement using the asteval Interpreter. Takes an expression
    containing parameter names and friend symbols recognizable by the asteval
    Interpreter.
    """
    return self._asteval.eval(expr)

  def __print__(self, oneline=False):
    """
    Return a pretty representation of a Parameters class.
    """
    if oneline:
      return super(Parameters, self).__repr__()
    s = "Parameters({\n"
    for key in self.keys():
        s += "    '%s': %s, \n" % (key, self[key])
    s += "    })\n"
    return s

  def __params_to_string(self, cols, col_offset):
    par_dict = {}
    len_dict = {}
    all_cols = ['name'] + cols
    for name, par in zip(self.keys(),self.values()):
      val, unc, pow = par.unc_round
      par_dict[name] = {}
      for col in all_cols:
        if col == 'name':
          par_dict[name][col] = getattr(par, col)
        elif col == 'value':
          if pow != '0':
            par_dict[name][col] = val+'e'+pow
          else:
            par_dict[name][col] = val
        elif col == 'stdev':
          if getattr(par, 'stdev'):
            if pow != '0':
              par_dict[name][col] = unc+'e'+pow
            else:
              par_dict[name][col] = unc
          else:
            par_dict[name][col] = 'None'
        elif col == 'brute_step':
          if getattr(par, 'stdev'):
            par_dict[name][col] = ".8f" % getattr(par, 'stdev')
          else:
            par_dict[name][col] = 'None'
        elif col == 'free':
          par_dict[name][col] = str(True == getattr(par, 'free'))
        elif col == 'min':
          par_dict[name][col] = str(getattr(par, 'min'))
        elif col == 'max':
          par_dict[name][col] = str(getattr(par, 'max'))
        elif col == 'latex':
          par_dict[name][col] = str(getattr(par, 'latex'))

    for col in all_cols:
      len_dict[col] = len(col) + col_offset
      for par in par_dict.values():
        len_dict[col] = max(len_dict[col], len(par[col]) + col_offset)
    return par_dict, len_dict

  def print(self, oneline=False,
                  cols=['value', 'stdev', 'min', 'max', 'free', 'latex'],
                  col_offset = 2):
    """
    Print parameters
    """
    if oneline: print(self.__print__(oneline=oneline)); return

    par_dict, len_dict = self.__params_to_string(cols, col_offset)
    # Formating line (will be used to print)
    line = '{:'+str(len_dict['name'])+'}'
    for col in cols[:-1]:
      line += ' {:>'+str(len_dict[col])+'}'
    line += '  {:'+str(len_dict['latex'])+'}\n'

    # Build the table
    all_cols = ['name'] + cols
    table = line.format(*all_cols).title()
    for name, par in zip(par_dict.keys(),par_dict.values()):
      table += line.format(*list(par.values()))
    print(table)



  def _repr_html_(self):
    """
    Returns a HTML representation of parameters data.
    """
    return params_html_table(self)



  def __add_parameter__(self, param):
    """
    Add a Parameter. If param is a Paramter then it will be directly stored in
    Parameters. If param is a dict, then a Parameter will be created and then
    stored.
    """
    if isinstance(param, Parameter):
      self.__setitem__(param.name, param)
    elif param:
      self.__setitem__(param['name'], Parameter(**param))
    else:
      raise KeyError("This is not a valid Parameter")



  def add(self, *params):
    """
    Add many parameters, using the given tuple.
    """
    for par in params: self.__add_parameter__(par)



  def valuesdict(self):
    """
    Dictionary (ordered one) of parameter values.
    """
    return OrderedDict((p.name, p.value) for p in self.values())



  def udict(self):
    """
    Dictionary (ordered one) of parameter values.
    """
    return OrderedDict((p.name, p.uvalue) for p in self.values())

  def dumps(self, **kws):
    """
    Prepare a JSON string of Parameters.
    """
    params = {p.name:p.__getstate__() for p in self.values()}
    return hjson.dumps(params, **kws)

  def loads(self, s, **kws):
    """
    Load Parameters from a JSON string (aka dict).
    """
    self.clear()
    #print(s)
    self.add(*tuple(s.values()))
    return self

  def dump(self, path, **kws):
    """
    Write JSON representation of Parameters to file given in path.
    """
    if path[:-4] != '.json': path += '.json'
    open(path,'w').write(self.dumps(**kws))
    return 0#open(path,'w').write(self.dumps(**kws))

  @classmethod
  def load(cls, path, **kws):
    """
    Load JSON representation of Parameters from a file given in path.
    """
    c = cls()
    c.loads(hjson.load(open(path,'r'), **kws))
    return c

  def update_constraints(self):
      """Update all constrained parameters, checking that dependencies are
      evaluated as needed."""
      requires_update = {name for name, par in self.items() if par._expr is
                         not None}
      updated_tracker = set(requires_update)

      def _update_param(name):
          """Update a parameter value, including setting bounds.

          For a constrained parameter (one with an `expr` defined),
          this first updates (recursively) all parameters on which the
          parameter depends (using the 'deps' field).

          """
          par = self.__getitem__(name)
          if par._expr_eval is None:
              par._expr_eval = self._asteval
          for dep in par._expr_deps:
              if dep in updated_tracker:
                  _update_param(dep)
          self._asteval.symtable[name] = par.value
          updated_tracker.discard(name)

      for name in requires_update:
          _update_param(name)

  def latex_dumps(self, cols=['latex', 'value', 'stdev', 'min', 'max', 'free'],
                        col_offset = 3,
                        split=False):
    """
    Print LaTeX parameters
    """
    par_dict, len_dict = self.__params_to_string(cols, col_offset)
    # Formating line (will be used to print)
    line = '${:'+str(len_dict['latex'])+'}$   '
    for col in cols[1:]:
      line += ' & ${:>'+str(len_dict[col])+'}$'
    line += '  \\\\ \n'

    # Build the table
    table  = "\\begin{table}\n\centering\n\\begin{tabular}{"+len(cols)*"c"+"}\n"
    table += "\hline\n"
    table += line.format(*cols).title() + '\hline\n'
    for name, par in zip(par_dict.keys(),par_dict.values()):
      table += line.format(*list(par.values())[1:])
    table += "\hline\n"
    table += "\end{tabular}\n\end{table}"
    print(table)
################################################################################










################################################################################
# Parameter ####################################################################

class Parameter(object):
  """
  A Parameter is an object that controls a model, it can be free or fixed \
  in a fit. A Parameter has several attributes to be completely described. \
  Those attributes are:
    * name: a valid string
    * value: A float number (default: 0)
    * free: True or False where the parameter if free or fixed (default: True)
    * min: Minimum value of the parameters (default:-inf)
    * max: Maximum value of the parameters (default:+inf)
    * expr: Mathematical expression used to constrain the value during the fit
    * brute_step: Step size for grid points in the (default: None)
    * init_value: Initial value for the fit (default: value),
    * correl: None,
    * stdev: None,
    * latex: LaTeX expression of the parameter name (default: name)

  Those atributes should be static they must exist always not depending of \
  the method used in the minimization.
  """


  def __init__(self, name=None, value=0, free=True, min=-inf, max=inf,
               expr=None, brute_step=None, user_data=None, init_value=None,
               correl=None, stdev=None, latex=None):
    """
    ---
    """
    self.name           = name
    self.latex          = name
    self.user_data      = user_data
    self.init_value     = value
    self.min            = min
    self.max            = max
    self.brute_step     = brute_step
    self.free           = free
    self.stdev          = stdev
    self._expr          = expr
    self.correl         = correl

    self._val           = value
    self._expr_ast      = None
    self._expr_eval     = None
    self._expr_deps     = []
    self._delay_asteval = False
    self._uvalue        = unc.ufloat(0,0)

    self.uncl           = self.stdev
    self.uncr           = self.stdev

    self.from_internal  = lambda val: val

    if latex: self.latex = latex
    if init_value: self.init_value = init_value

    self._init_bounds()



  def set(self, value=None,
                free=None,
                min=None,
                max=None,
                expr=None,
                brute_step=None):
    """
    Update Parameter attributes.

    In:
    0.123456789:
          value:  New float number
           free:  True or False
            min:  To remove limits use '-inf', not 'None'
            max:  To remove limits use '+inf', not 'None'
           expr:  To remove a constraint you must supply an empty string ''
     brute_step:  To remove the step size you must use '0'
     init_value:  Initial value for the fit (default: value),
         correl:  None,
          stdev:  None,
          latex:  LaTeX expression of the parameter name (default: name)

    Out:
           void
    """
    if value is not None:
      self.value = value
      self.__set_expression('')

    if free is not None:
      self.free = free
      if free:
        self.__set_expression('')

    if min is not None:
      self.min = min

    if max is not None:
      self.max = max

    if expr is not None:
      self.__set_expression(expr)

    if brute_step is not None:
      if brute_step == 0.0:
        self.brute_step = None
      else:
        self.brute_step = brute_step



  def _init_bounds(self):
    """
    Make sure initial bounds are self-consistent.
    """
    # _val is None means - infinity.
    if self.max is None:
      self.max = inf
    if self.min is None:
      self.min = -inf
    if self._val is None:
      self._val = -inf
    if self.min > self.max:
      self.min, self.max = self.max, self.min
    if isclose(self.min, self.max, atol=1e-13, rtol=1e-13):
      raise ValueError("Parameter '%s' has min == max" % self.name)
    if self._val > self.max:
      self._val = self.max
    if self._val < self.min:
      self._val = self.min
    self.setup_bounds()

  def __getstatepickle__(self):
      """Get state for pickle."""
      return (self.name, self.value, self.free, self.expr, self.min,
              self.max, self.brute_step, self.stdev, self.correl,
              self.init_value, self.user_data)

  def __getstate__(self):
    """
    Get state for json.
    """
    return {"name":self.name, "value":self.value, "free":self.free,
            "expr":self.expr, "min":self.min, "max": self.max,
            "brute_step": self.brute_step, "stdev":self.stdev,
            "correl":self.correl, "init_value":self.init_value,
            "user_data":self.user_data, "latex":self.latex}



  def __setstate__(self, state):
    """
    Set state for pickle. ¿IS THIS NEEDED?
    """
    (self.name, self.value, self.free, self.expr, self.min, self.max,
     self.brute_step, self.stdev, self.correl, self.init_value,
     self.user_data) = state
    self._expr_ast = None
    self._expr_eval = None
    self._expr_deps = []
    self._delay_asteval = False
    self._init_bounds()



  def __repr__(self):
    """
    Return the representation of a Parameter object.
    """
    s = []
    if self.name is not None:
        s.append("'%s'" % self.name)
    sval = repr(self._getval())
    if not self.free and self._expr is None:
      sval = "value=%s (fixed)" % sval
    elif self.stdev is not None:
      sval = "value=%s +/- %.3g (free)" % (sval, self.stdev)
    else:
      sval = "value=%s (free)" % sval
    s.append(sval)
    s.append("limits=[%s:%s]" % (repr(self.min), repr(self.max)))
    if self._expr is not None:
        s.append("expr='%s'" % self.expr)
    if self.brute_step is not None:
        s.append("brute_step=%s" % (self.brute_step))
    return "<Parameter %s>" % ', '.join(s)



  def setup_bounds(self):
      """Set up Minuit-style internal/external parameter transformation of
      min/max bounds.

      As a side-effect, this also defines the self.from_internal method
      used to re-calculate self.value from the internal value, applying
      the inverse Minuit-style transformation. This method should be
      called prior to passing a Parameter to the user-defined objective
      function.

      This code borrows heavily from JJ Helmus' leastsqbound.py

      Returns
      -------
      _val : float
          The internal value for parameter from self.value (which holds
          the external, user-expected value). This internal value should
          actually be used in a fit.

      """
      if self.min is None:
          self.min = -inf
      if self.max is None:
          self.max = inf
      if self.min == -inf and self.max == inf:
          self.from_internal = lambda val: val
          _val = self._val
      elif self.max == inf:
          self.from_internal = lambda val: self.min - 1.0 + sqrt(val*val + 1)
          _val = sqrt((self._val - self.min + 1.0)**2 - 1)
      elif self.min == -inf:
          self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
          _val = sqrt((self.max - self._val + 1.0)**2 - 1)
      else:
          self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                               (self.max - self.min) / 2.0
          _val = arcsin(2*(self._val - self.min)/(self.max - self.min) - 1)
      return _val



  def scale_gradient(self, val):
    """Return scaling factor for gradient.

    Parameters
    ----------
    val: float
        Numerical Parameter value.

    Returns
    -------
    float
        Scaling factor for gradient the according to Minuit-style
        transformation.

    """
    if self.min == -inf and self.max == inf:
        return 1.0
    elif self.max == inf:
        return val / sqrt(val*val + 1)
    elif self.min == -inf:
        return -val / sqrt(val*val + 1)
    return cos(val) * (self.max - self.min) / 2.0



  def _getval(self):
    """Get value, with bounds applied."""
    # Note assignment to self._val has been changed to self.value
    # The self.value property setter makes sure that the
    # _expr_eval.symtable is kept updated.
    # If you just assign to self._val then
    # _expr_eval.symtable[self.name]
    # becomes stale if parameter.expr is not None.
    if (isinstance(self._val, uncertainties.core.Variable) and
            self._val is not nan):

        try:
            self.value = self._val.nominal_value
        except AttributeError:
            pass
    if not self.free and self._expr is None:
        return self._val

    if self._expr is not None:
        if self._expr_ast is None:
            self.__set_expression(self._expr)

        if self._expr_eval is not None:
            if not self._delay_asteval:
                self.value = self._expr_eval(self._expr_ast)
                check_ast_errors(self._expr_eval)

    if self._val is not None:
        if self._val > self.max:
            self._val = self.max
        elif self._val < self.min:
            self._val = self.min
    if self._expr_eval is not None:
        self._expr_eval.symtable[self.name] = self._val
    return self._val

  def set_expr_eval(self, evaluator):
    """
    Set expression evaluator instance.
    """
    self._expr_eval = evaluator

  @property
  def value(self):
    """
    Return the numerical value of the Parameter, with bounds applied.
    """
    return self._getval()



  @value.setter
  def value(self, val):
    """
    Set the numerical Parameter value.
    """
    self._val = val
    if not hasattr(self, '_expr_eval'):
      self._expr_eval = None
    if self._expr_eval is not None:
      self._expr_eval.symtable[self.name] = val



  @property
  def expr(self):
    """
    Return the mathematical expression used to constrain the value
    during the fit.
    """
    return self._expr



  @expr.setter
  def expr(self, val):
    """
    Set the mathematical expression used to constrain the value during
    the fit.

    To remove a constraint you must supply an empty string.

    """
    self.__set_expression(val)



  @property
  def uvalue(self):
    change = 0
    if self._uvalue.n != self.value:
      change = 1
    if self._uvalue.s == 0.0:
      if (self.stdev != 0.0) | (self.stdev is not None):
        change = 1
    if change:
      if self.stdev:
        self._uvalue = unc.ufloat(self.value,self.stdev)
      else:
        self._uvalue = unc.ufloat(self.value,0)
    # print(id(self._uvalue)) # for checking if something is changing...
    return self._uvalue



  def dumps_latex(self):
    return self.name + ' = ' + '{:.2uL}'.format(self.uvalue)


  @property
  def unc_round(self):
    par_str = '{:.2uL}'.format(self.uvalue)
    #par_str = "\left(2.00 \pm 0.10\right) \times 10^{6}"
    #par_str = "2.00 \pm 0.10"
    if len(par_str.split(r'\times 10^')) > 1:
      expr, pow = par_str.split(r'\times 10^')
      expr = expr.split(r'\left(')[1].split('\right)')[0]
      pow = pow.split('{')[1].split('}')[0]
    else:
      expr = par_str; pow = '0'
    return expr.split(r' \pm ')+[pow]



  def __set_expression(self, val):
    if val == '':
      val = None
    self._expr = val
    if val is not None:
      self.free = False
    if not hasattr(self, '_expr_eval'):
      self._expr_eval = None
    if val is None:
      self._expr_ast = None
    if val is not None and self._expr_eval is not None:
      self._expr_eval.error = []
      self._expr_eval.error_msg = None
      self._expr_ast = self._expr_eval.parse(val)
      check_ast_errors(self._expr_eval)
      self._expr_deps = get_ast_names(self._expr_ast)



  # Define common operations over parameters -----------------------------------
  def __array__(self): return array(float(self.uvalue))

  def __str__(self): return self.__repr__()

  def __abs__(self): return abs(self.uvalue)

  def __neg__(self): return -self.uvalue

  def __pos__(self): return +self.uvalue

  def __nonzero__(self): return self.uvalue != 0

  def __int__(self): return int(self.uvalue)

  def __float__(self): return float(self.uvalue)

  def __trunc__(self): return self.uvalue.__trunc__()

  def __add__(self, friend): return self.uvalue + friend

  def __sub__(self, friend): return self.uvalue - friend

  def __div__(self, friend): return self.uvalue / friend

  def __floordiv__(self, friend): return self.uvalue // friend

  def __divmod__(self, friend): return divmod(self.uvalue, friend)

  def __mod__(self, friend): return self.uvalue % friend

  def __mul__(self, friend): return self.uvalue * friend

  def __pow__(self, friend): return self.uvalue ** friend

  def __gt__(self, friend): return self.uvalue > friend

  def __ge__(self, friend): return self.uvalue >= friend

  def __le__(self, friend): return self.uvalue <= friend

  def __lt__(self, friend): return self.uvalue < friend

  def __eq__(self, friend): return self.uvalue == friend

  def __ne__(self, friend): return self.uvalue != friend

  def __radd__(self, friend): return friend + self.uvalue

  def __rdiv__(self, friend): return friend / self.uvalue

  def __rdivmod__(self, friend): return divmod(friend, self.uvalue)

  def __rfloordiv__(self, friend): return friend // self.uvalue

  def __rmod__(self, friend): return friend % self.uvalue

  def __rmul__(self, friend): return friend * self.uvalue

  def __rpow__(self, friend): return friend ** self.uvalue

  def __rsub__(self, friend): return friend - self.uvalue


# Paramter-ness checher --------------------------------------------------------
def isParameter(x):
    """
    Check if an object belongs to Parameter-class.
    """
    return (isinstance(x, Parameter) or x.__class__.__name__ == 'Parameter')



################################################################################
