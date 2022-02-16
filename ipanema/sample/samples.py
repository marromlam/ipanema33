################################################################################
#                                                                              #
#                                   SAMPLES                                    #
#                                                                              #
################################################################################

import numpy as np
import pandas
import os
import json

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import uproot3 as uproot
import re
import builtins

from ..params import Parameters
from ..core.utils import ristra

from ..tools.misc import get_vars_from_string


def cuts_and(*args):
  result = list(args)
  result = [a for a in args if a]
  return '(' + ') & ('.join(result) + ')'

__all__ = ['cuts_and', 'get_vars_from_string', 'get_data_file', 'Sample', 'Categories', 'DataSet']

################################################################################
# Function to parse config files ###############################################

def get_data_file(file_path):
  file = json.load(open(file_path))
  # New Functions
  def alpha(x, y=1):
    z = x/y
    return z*( (z.sum())/((z**2).sum()) )
  needed_vars = []
  data = uproot.open(file['path'])[file['tree_name']]
  input_vars = data.keys()
  for var in file['branches'].values():
    new_ones = get_vars_from_string(var)
    needed_vars += [new for new in new_ones if new.encode() in input_vars]
  data = data.pandas.df(needed_vars)
  #print(needed_vars)
  if file['cuts']:
    data = data.query(file['cuts'])
  output_df = pandas.DataFrame()
  for var in file['branches'].keys():
    try:
      output_df[var] = data.eval(file['branches'][var])
    except:
      #print('@'+file['branches'][var])
      output_df[var] = data.eval('@'+file['branches'][var],engine='python')
  return output_df

################################################################################



################################################################################
################################################################################

class Categories(object):
  """docstring for Categories."""

  def __init__(self, arg):
    super(Categories, self).__init__()
    self.arg = arg

################################################################################



################################################################################
################################################################################

class DataSet(object):
  """docstring for Categories."""

  def __init__(self, cats=None, name='untitled DataSet',
              cuts = None, params = None,
               copy=True, convert=True, trim=False, backup=False, path=None,
               verbose=True):
    self._name = name
    self._params = Parameters()
    self._categories = {}
    self._cuts = ""

    if params:
      self._params = params
    if cats:
      for catname,cat in cats.items():
        self.add_params(catname,cat)

    if verbose:
      print(f"{'ipanema.sample':>15}: {name}")
      print(f"{'from':>15}: {path}")
      print(f"{'categories':>15}: {list(self._categories.keys())}")
      print(f"{'cuts':>15}: {cuts}")

  def __iter__(self):
     return iter(self._categories.values())

  def category(self, catname):
    return self._categories[catname]



  def add_category(self, *cats):
    # check if cat is ipanema.Sample object
    for cat in cats:
      if isSample(cat):
        self._categories.update({cat.name:cat})
      else:
        raise ValueError("Provided cat is not a valid ipanema.Sample object")



  def rm_category(self, catname):
    self._categories.pop(catname)



  def add_params(self, params):
    self._params = Parameters.load(params)



  @property
  def params(self):
    return self._params



  @property
  def categories(self):
    return list(self._categories.keys())

  def find_categories(self, word):
    regex = re.compile(word)
    all_cats = self._categories.keys()
    return [ key for key in all_cats if regex.match(key) ]


  def split(self, mothercat, cut, mothername=None, childnames=[None,None],
            copy=True, convert=True, trim=False):
    name = mothercat
    if mothername:
      name = f"{name}_{mothername}"
      self._categories[mothercat].name = mothername

    if not childnames[0]:
      childnames[0] = f"{name}_({cut})false"
    if not childnames[1]:
      childnames[1] = f"{name}_({cut})true"

    tsample = Sample.from_pandas(self._categories[mothercat].df, cuts = cut,
                                 name=f"{mothercat}_{childnames[1]}",
                                 copy=copy, convert=convert, trim=trim)
    fsample = Sample.from_pandas(self._categories[mothercat].df, cuts = f"~({cut})",
                                 name=f"{mothercat}_{childnames[0]}",
                                 copy=copy, convert=convert, trim=trim)
    dsample = Sample.from_pandas(self._categories[mothercat].df,
                                 name=f"{name}",
                                 copy=copy, convert=convert, trim=trim)


    self.add_category(tsample,fsample,dsample)
    if mothername:
      self.rm_category(mothercat)



################################################################################








################################################################################
################################################################################

class Sample(object):
  """
  docstring for Sample.
  """

  def __init__(self, df, name='untitled', cuts = None, params = None,
               copy=True, convert=True, trim=False, backup=False, path=None,
               verbose=False):
    self.name = name
    self.__backup = backup
    if self.__backup:
      self.__df = df                               # to maintain an orginal copy
    self._cuts = ""

    self.df = df
    if cuts:
      self.chop(cuts)

    self.params = params
    self.path = path

    if verbose:
      print(self.__str__())

  def __str__(self):
    table = []
    table.append(f"{'ipanema.sample':>15}: {self.name}")
    table.append(f"{'from':>15}: {self.path}")
    table.append(f"{'size':>15}: {self.shape}")
    table.append(f"{'cuts':>15}: {self._cuts}")
    return "\n".join(table)

  def __get_name(self, filename):
    namewithextension = os.path.basename(os.path.normpath(filename))
    return os.path.splitext(namewithextension)[0]

  @property
  def branches(self):
    return list(self.df.keys())

  @property
  def shape(self):
    return self.df.shape

  def find(self, word):
    branches = self.branches
    regex = re.compile(word)
    return [ b for b in branches if regex.match(b) ]

  @classmethod
  def from_file(cls, filename, name = None, cuts = None, params = None,
                copy=True, convert=True, trim=False):
    if filename[-5:] != '.json': filename += '.json'
    if not name:
      namewithextension = os.path.basename(os.path.normpath(filename))
      name = os.path.splitext(namewithextension)[0]
    return cls(get_data_file(filename), name, cuts = cuts, params = params,
               copy=True, convert=True, trim=False, path=filename)

  @classmethod
  def from_pandas(cls, df, name = None, cuts = None, params = None,
                  copy=True, convert=True, trim=False):
    return cls(df, name, cuts=cuts, params=params,
               copy=copy, convert=convert, trim=trim, path=None)


  @classmethod
  def from_numpy(cls, arrdicts, name = None, cuts = None, params = None,
                  copy=True, convert=True, trim=False):
    df = pandas.DataFrame.from_dict(arrdicts)
    return cls(df, name, cuts=cuts, params=params,
               copy=copy, convert=convert, trim=trim, path=None)

  @classmethod
  def from_root(cls, filename, treename='DecayTree', name = None, cuts = None,
                params=None, copy=True, convert=True, trim=False, share=False, backup=False,
                branches=None, **up_kwgs):
    if filename[-5:] != '.root': filename += '.root'
    if not name:
      namewithextension = os.path.basename(os.path.normpath(filename))
      name = os.path.splitext(namewithextension)[0]
    if share:
      # uproot4 current workaround: read fisrt branche and get len there
      #b0 = uproot.open(filename)[treename].keys()[0]
      #num_entries = len(uproot.open(filename)[treename][b0].array())
      #noe = round(num_entries*share/100)
      #up_kwgs.update(max_num_elements=noe)
      noe = round(uproot.open(filename)[treename]._fEntries*share/100)
      up_kwgs.update(entrystop=noe)
    # uproot4: df = uproot.open(filename,**up_kwgs)[treename].arrays(branches, library="pd")
    df = uproot.open(filename)[treename].pandas.df(branches=branches, **up_kwgs)
    return cls(df, name, cuts=cuts, params=params,
               copy=copy, convert=convert, trim=trim, backup=backup, path=filename)

  def add(self, name, attribute):
    self.__setattr__(name,attribute)

  def cut(self, cut = None):
    """
    Place cuts on df and return it with them applied!
    """
    if cut:
      self._cuts = cuts_and(cut, self._cuts)
      return self.df.query(cut)
    return self.df

  def chop(self, cut = None):
    """
    Place cuts on df and actually cut df
    """
    self.df = self.cut(cut)

  def back_to_original(self):
    """
    If there is a backup df, load it
    """
    if self.__backup:
      self.df = self.__df
    else:
      print('There is no avaliable backup. DataFrame remains the same.')

  def allocate(self, **branches):
    """
    Creates a property called by provided key
    """
    for var, expr in zip(branches.keys(),branches.values()):
      if isinstance(expr, str):
        this_branch = np.ascontiguousarray(self.df.eval(expr).values)
      else:
        this_branch = []
        for item in expr:
          this_branch.append(np.array(self.df.eval(item).values))
        this_branch = tuple(this_branch)
        this_branch = np.ascontiguousarray(np.stack(this_branch, axis=-1))
      self.add(var, ristra.allocate(this_branch).astype(np.float64))

  def assoc_params(self, params):
    self.params = Parameters.load(params)

################################################################################


def isSample(x):
  """
  Check if an object belongs to Sample-class.
  """
  return (isinstance(x, Sample) or x.__class__.__name__ == 'Sample')
