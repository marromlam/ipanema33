################################################################################
#                                                                              #
#                           PARAMETER & PARAMETERS                             #
#                                                                              #
################################################################################

from collections import OrderedDict
import hjson
from numpy import arcsin, array, cos, inf, isclose, nan, sin, sqrt
from numpy import inf as Infinite
import uncertainties as unc
import re
import numpy as np

# Parameter formula stuff
from asteval import Interpreter, get_ast_names, valid_symbol_name
import scipy.special

AST_FUNCTIONS_DICT = {} # Get some functions from scipy to be handled by asteval
for name in ['gamma', 'erf', 'erfc', 'wofz']:
  AST_FUNCTIONS_DICT['sc_'+name] = getattr(scipy.special, name)

# Asteval error checker
def _check_ast_errors_(formula_eval):
  if len(formula_eval.error) > 0: formula_eval.raise_exception(None)



################################################################################
# Parameters ###################################################################

class Parameters(OrderedDict):
  """
  An ordered dictionary of all the Parameter objects. Note:
  * Parameter().name must be a valid Python symbol name
  * Parameters() is made only of Parameter() items
  """

  def __init__(self, asteval=None, usersyms=None, *args, **kwargs):
    super(Parameters, self).__init__(self)

    self._asteval = asteval
    if self._asteval is None:
      self._asteval = Interpreter()
    _syms = {}
    _syms.update(AST_FUNCTIONS_DICT)
    if usersyms is not None:
      _syms.update(usersyms)
    for key, val in _syms.items():
      self._asteval.symtable[key] = val
    self.update(*args, **kwargs)



  def copy(self, params_in):
    """
    Alias of __copy__.
    """
    return self.__copy__(params_in)


  @classmethod
  def clone(cls, params_in):
    """
    Alias of __copy__.
    """
    return cls().__deepcopy__(params_in)


  @classmethod
  def __deepcopy__(cls, params_in):
    c = cls()
    c.loads(hjson.loads(params_in.dumps()))
    return c



  def __copy__(self, params_in):
    self.loads(hjson.loads(params_in.dumps()))
    return self



  def __setitem__(self, key, par):
    if key not in self:
      if not valid_symbol_name(key):
        raise KeyError("'%s' has not a valid Parameter name" % key)
    if par is not None and not isinstance(par, Parameter):
      raise ValueError("'%s' is not a Parameter" % par)
    OrderedDict.__setitem__(self, key, par)
    par.name = key
    par._formula_eval_ = self._asteval
    self._asteval.symtable[key] = par.value



  def __add__(self, friend):
    """
    Merge Parameters objects.
    """
    if not isinstance(friend, Parameters):
      raise ValueError("'%s' is not a Parameters object" % friend)
    out = self.__deepcopy__(self)
    pars_original = list(out.keys())
    pars_friend = list(friend.keys())
    for par in pars_friend:
      if par not in pars_original:
        out.add(friend[par])
    return out



  def __array__(self):
    """
    Convert Parameters to array.
    """
    try:
      arr = array([float(k) for k in self.values()])
    except:
      arr = array([float(k.value) for k in self.values()])
    finally:
      return arr



  def eval(self, formula):
    """
    Evaluate a statement using the asteval Interpreter. Takes an expression
    containing parameter names and friend symbols recognizable by the asteval
    Interpreter.
    """
    return self._asteval.eval(formula)



  def find(self, word):
    regex = re.compile(word)
    list_parameters = list( self.keys() )
    return [ key for key in list_parameters if regex.match(key) ]


  def fetch(self, params):
    if isinstance(params,str):
      params = self.find(params)
    return { k:self[k] for k in params }



  @classmethod
  def build(cls, params, params_list):
    c = cls(); temp = cls()
    for k in params_list:
      temp.add(params[k])
    return c.__deepcopy__(temp)

  # @classmethod
  # def __deepcopy__(cls, params_in):
  #   c = cls()
  #   c.loads(hjson.loads(params_in.dumps()))
  #   return c

  def __str__(self, cols=['value', 'stdev', 'min', 'max', 'free'], col_offset = 2):
    """
    Return a pretty representation of a Parameters class.
    """
    par_dict, len_dict = self._params_to_string_(cols, col_offset)

    # Formating line (will be used to print)
    line = '{:'+str(len_dict['name'])+'}'
    for col in cols[:-1]:
      line += ' {:>'+str(len_dict[col])+'}'
    line += '  {:'+str(len_dict[cols[-1]])+'}\n'

    # Build the table
    all_cols = ['name'] + cols
    table = line.format(*all_cols).title()
    for name, par in zip(par_dict.keys(),par_dict.values()):
      table += line.format(*list(par.values()))
    return table



  def _params_to_string_(self, cols, col_offset):
    """
    Prepare strings of parameters to be printed. This function is used both
    to print parameters and to dump them to LaTeX.
    """
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
        elif col == 'reldev':
          if getattr(par, 'stdev'):
            par_dict[name][col] = f"({abs(par.stdev/par.value):.2%})"
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



  def print(self, cols=['value', 'stdev', 'min', 'max', 'free', 'latex'], col_offset = 2, as_string = False):
    """
    Print parameters table
    """
    table = self.__str__(cols,col_offset)
    if as_string:
      return table
    print(table)



  def _add_parameter_(self, param):
    """
    Add a Parameter. If param is a Parameter then it will be directly stored in
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
    for par in params: self._add_parameter_(par)



  def valuesdict(self,blind=True):
    """
    OrderedDict of parameter values.
    """
    return OrderedDict((p.name, p._getval(blind)) for p in self.values())



  def uvaluesdict(self):
    """
    OrderedDict of parameter values.
    """
    return OrderedDict((p.name, p.uvalue) for p in self.values())


  def correl_mat(self):
    pars = list(self.keys())
    corr_mat = np.zeros((len(pars),len(pars)))
    for i in range(0,len(pars)):
      p = pars[i]
      corr = self[p].correl
      for j in range(0,len(pars)):
        if pars[j] in corr:
          corr_mat[i][j] = corr[pars[j]]
        else:
          corr_mat[i][j] = np.nan
    return corr_mat


  def lock(self,*args):
    if args:
      for par in args:
        self[par].free = False
    else:
      for par in self:
        self[par].free = False



  def unlock(self,*args):
    if args:
      for par in args:
        self[par].free = True
    else:
      for par in self:
        self[par].free = True



  def dumps(self, **kwargs):
    """
    Prepare a JSON string of Parameters.
    """
    params = {p.name:p.__getstate__() for p in self.values()}
    for p in params:
      filter = {k: v for k, v in params[p].items() if v is not None}
      params[p].clear()
      params[p].update(filter)
    return hjson.dumps(params, **kwargs)



  def loads(self, s, **kwargs):
    """
    Load Parameters from a JSON string (aka dict).
    """
    self.clear()
    self.add(*tuple(s.values()))
    return self



  def dump(self, path, **kwargs):
    """
    Write JSON representation of Parameters to file given in path.
    """
    if path[-5:] != '.json':
      path += '.json'
    open(path,'w').write(self.dumps(**kwargs))


  @classmethod
  def load(cls, path, **kwargs):
    """
    Load JSON representation of Parameters from a file given in path.
    """
    c = cls()
    c.loads(hjson.load(open(path,'r'), **kwargs))
    return c



  def update_constraints(self):
    """
    Update all constrained parameters, checking that dependencies are
    evaluated as needed.
    """
    requires_update = {name for name, par in self.items() if par._formula is not None}
    updated_tracker = set(requires_update)

    def _update_param_(name):
      """
      Update a parameter value, including setting bounds.

      For a constrained parameter (one with an `formula` defined),
      this first updates (recursively) all parameters on which the
      parameter depends (using the 'deps' field).
      """
      par = self.__getitem__(name)
      if par._formula_eval_ is None:
        par._formula_eval_ = self._asteval
      for dep in par._formula_deps:
        if dep in updated_tracker:
            _update_param_(dep)
      self._asteval.symtable[name] = par.value
      updated_tracker.discard(name)

    for name in requires_update:
      _update_param_(name)



  def dump_latex(self, cols=['value','stdev'], col_offset=3, caption='None',
                 verbose=False):
    """
    Print LaTeX parameters

    TODO: I think when some parameter value has 10^exp will be represented as
          1eexp. Some mod is needed to rewrite that e into \times 10^exp. :)
    """
    cols = ['latex'] + cols
    par_dict, len_dict = self._params_to_string_(cols, col_offset)

    # Formating line (will be used to print)
    line = '${:'+str(len_dict['latex'])+'}$   '
    for col in cols[1:]:
      line += ' & ${:>'+str(len_dict[col])+'}$'
    line += '  \\\\ \n'

    # Build the table
    table  = "\\begin{table}\n\centering\n\\begin{tabular}{"+len(cols)*"c"+"}\n"
    table += "\hline\n"
    table += line.format(*cols).title().replace('$',' ') + '\hline\n'
    for name, par in zip(par_dict.keys(),par_dict.values()):
      table += line.format(*list(par.values())[1:])
    table += "\hline\n\end{tabular}\n"
    table += f"\caption{{{caption}}}\n"
    table += "\end{table}\n"
    table = table.replace('None','    ')
    table = table.replace('Latex    ','Parameter')
    if verbose:
      print(table)
    return table

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
    * formula: Mathematical formula used to constrain the value during the fit
    * init: Initial value for the fit (default: value),
    * correl: None,
    * stdev: None,
    * latex: LaTeX formula of the parameter name (default: name)

  Those atributes should be static they must exist always not depending of \
  the method used in the minimization.
  """


  def __init__(self, name=None, value=0, free=True, min=-inf, max=inf,
               formula=None, casket=None, init=None,
               correl=None, stdev=None, latex=None,
               blind=None, blindscale = 1, blindengine='python'):
    """
    Object that controls a model

    In:
    0.123456789:
           name:  Parameter's name.
                  string
          value:  Parameter's value.
                  float (default: 0)
           free:  Whether the Parameter can vary or not during a fit.
                  bool (default: True)
            min:  Minimum value of the Parameter's range.
                  float (default:-inf)
            max:  Maximum value of the Parameter's range.
                  float (default:+inf)
        formula:  Mathematical formula used to constrain the value during the fit.
                  string (default=None)
           init:  Initial value for the fit,
                  float (default: value)
         correl:  Correlation
                  float (default=None)
          stdev:  Parameter standard deviation.
                  float, (default=None)
          latex:  LaTeX expression of Parameter's name.
                  string, (default: name)

    Out:
           void

    """
    self.name           = name
    self.latex          = name

    self.init           = value
    self.min            = min
    self.max            = max
    self.free           = free
    self.stdev          = stdev
    self.correl         = correl

    self._formula       = formula
    self._value         = value
    self._formula_ast   = None
    self._formula_eval_ = None
    self._formula_deps  = []
    self._delay_asteval = False
    self._uvalue        = unc.ufloat(0,0)

    self.casket         = casket
    self.uncl           = self.stdev
    self.uncr           = self.stdev

    self.from_internal  = lambda val: val

    if latex: self.latex = latex
    if init: self.init = init
    self.blind = blind
    self.__blinding__ = 0
    if blind:
      if blindengine=='python':
        np.random.seed( abs(hash('blindstr')//(2**32-1)) )
        self.__blinding__ = (value-blindscale)+blindscale*np.random.rand()
      elif blindengine=='root':
        print('shit not implemented yet')

    self._check_init_bounds_()



  def set(self, value=None, free=None, min=None, max=None, formula=None):
    """
    Update Parameter attributes.

    In:
    0.123456789:
          value:  New float number
           free:  True or False
            min:  To remove limits use '-inf', not 'None'
            max:  To remove limits use '+inf', not 'None'
        formula:  To remove a constraint you must supply an empty string ''
           init:  Initial value for the fit (default: value),
         correl:  None,
          stdev:  None,
          latex:  LaTeX formula of the parameter name (default: name)

    Out:
           void
    """
    self.__setstate__(value, free, min, max, formula)



  def _check_init_bounds_(self):
    """
    Make sure initial bounds are self-consistent.
    """
    # _value is None means - infinity.
    if self.max is None:
      self.max = inf
    if self.min is None:
      self.min = -inf
    if self.min > self.max:
      self.min, self.max = self.max, self.min
    if isclose(self.min, self.max, atol=1e-14, rtol=1e-14):
      self.free = False
    if self._value is None:
      self._value = self.min
    if self._value > self.max:
      self._value = self.max
    if self._value < self.min:
      self._value = self.min
    self.setup_bounds()



  def __setstate__(self, value=None, free=None, min=None, max=None, formula=None):
    if value is not None:
      self._value = value
      self._set_formula_('')

    if free is not None:
      self.free = free
      if free:
        self._set_formula_('')

    if min is not None:
      self.min = min

    if max is not None:
      self.max = max

    if formula is not None:
      self._set_formula_(formula)


  def __getstate__(self):
    """
    Get state for json.
    """
    return {"name":self.name, "value":self.value, "free":self.free,
            "formula":self.formula, "min":self.min, "max": self.max,
            "stdev":self.stdev, "correl":self.correl, "init":self.init,
            "casket":self.casket, "latex":self.latex, "blind":self.blind}



  def __repr__(self):
    """
    Return the representation of a Parameter object.
    """
    s = []
    if self.name is not None:
        s.append("'%s'" % self.name)
    #sval = repr(self._getval())
    sval = repr(self.value)
    if not self.free and self._formula is None:
      sval = "value=%s (fixed)" % sval
    elif self.stdev is not None:
      sval = "value=%s +/- %.3g (free)" % (sval, self.stdev)
    else:
      sval = "value=%s (free)" % sval
    s.append(sval)
    s.append("limits=[%s:%s]" % (repr(self.min), repr(self.max)))
    if self._formula is not None:
        s.append("formula='%s'" % self.formula)
    if self.blind:
        s.append("blinded")
    return "<Parameter %s>" % ', '.join(s)



  def setup_bounds(self):
    """
    Set up Minuit-style internal/external parameter transformation of
    min/max bounds. This was taken from JJ Helmus' leastsqbound.py.

    """
    #print(self)
    if self.min is None:
      self.min = -inf
    if self.max is None:
      self.max = inf
    if self.min == -inf and self.max == inf:
      self.from_internal = lambda val: val
      _value = self._value
    elif self.max == inf:
      self.from_internal = lambda val: self.min - 1.0 + sqrt(val*val + 1)
      _value = sqrt((self._value - self.min + 1.0)**2 - 1)
    elif self.min == -inf:
      self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
      _value = sqrt((self.max - self._value + 1.0)**2 - 1)
    else:
      self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                            (self.max - self.min) / 2.0
      _value = arcsin(2*(self._value - self.min)/(self.max - self.min) - 1)
      _value = self.init
    return _value



  def scale_gradient(self, value):
    """
    Minuit-style transformation for the gradient scaling factor.

    In:
    0.123456789:
          value:  Value
                  float

    Out:
              0:  Scaling factor
                  float

    """
    if self.min == -inf and self.max == inf:
      return 1.0
    elif self.max == inf:
      return value / sqrt(value*value + 1)
    elif self.min == -inf:
      return -value / sqrt(value*value + 1)
    return cos(value) * (self.max - self.min) / 2.0



  def _getval(self, unblind=False):
    """Get value, with bounds applied."""
    # Note assignment to self._value has been changed to self.value
    # The self.value property setter makes sure that the
    # _formula_eval_.symtable is kept updated.
    # If you just assign to self._value then
    # _formula_eval_.symtable[self.name]
    # becomes stale if parameter.formula is not None.
    if (isinstance(self._value, unc.core.Variable) and
            self._value is not nan):

        try:
            self.value = self._value.nominal_value
        except AttributeError:
            pass
    if not self.free and self._formula is None:
        return self._value

    if self._formula is not None:
        if self._formula_ast is None:
            self._set_formula_(self._formula)

        if self._formula_eval_ is not None:
            if not self._delay_asteval:
                self.value = self._formula_eval_(self._formula_ast)
                _check_ast_errors_(self._formula_eval_)

    if self._value is not None:
        if self._value > self.max:
            self._value = self.max
        elif self._value < self.min:
            self._value = self.min
    if self._formula_eval_ is not None:
        self._formula_eval_.symtable[self.name] = self._value
    if not unblind:
        #print('yeah, im unblinded to you')
        return self._value + self.__blinding__
    return self._value



  def set_formula_eval_(self, evaluator):
    """
    Set formula evaluator instance.
    """
    self._formula_eval_ = evaluator



  @property
  def value(self):
    """
    Return the numerical value of the Parameter, with bounds applied.
    """
    #print('yeassss')
    return self._getval(True)



  @value.setter
  def value(self, val):
    """
    Set the numerical Parameter value.
    """
    self._value = val
    if not hasattr(self, '_formula_eval_'):
      self._formula_eval_ = None
    if self._formula_eval_ is not None:
      self._formula_eval_.symtable[self.name] = val



  @property
  def formula(self):
    """
    Return the mathematical formula used to constrain the value
    during the fit.
    """
    return self._formula



  @formula.setter
  def formula(self, val):
    """
    Set the mathematical formula used to constrain the value during
    the fit.

    To remove a constraint you must supply an empty string.

    """
    self._set_formula_(val)



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



  def dump_latex(self):
    # Return a parameter.latex = value+/-stdev
    return self.latex + ' = ' + '{:.2uL}'.format(self.uvalue)



  @property
  def unc_round(self):
    par_str = '{:.2uL}'.format(self.uvalue)
    if len(par_str.split(r'\times 10^')) > 1:
      formula, pow = par_str.split(r'\times 10^')
      formula = formula.split(r'\left(')[1].split(r'\right)')[0]
      pow = pow.split('{')[1].split('}')[0]
    else:
      formula = par_str; pow = '0'
    return formula.split(r' \pm ')+[pow]



  def _set_formula_(self, val):
    if val == '':
      val = None
    self._formula = val
    if val is not None:
      self.free = False
    if not hasattr(self, '_formula_eval_'):
      self._formula_eval_ = None
    if val is None:
      self._formula_ast = None
    if val is not None and self._formula_eval_ is not None:
      self._formula_eval_.error = []
      self._formula_eval_.error_msg = None
      self._formula_ast = self._formula_eval_.parse(val)
      _check_ast_errors_(self._formula_eval_)
      self._formula_deps = get_ast_names(self._formula_ast)



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


# Parameter-ness checher -------------------------------------------------------

def isParameter(x):
  """
  Check if an object belongs to Parameter-class.
  """
  return (isinstance(x, Parameter) or x.__class__.__name__ == 'Parameter')



################################################################################
