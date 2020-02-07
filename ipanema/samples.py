import numpy as np
import pandas
import os
import json
import uproot
#import pycuda.driver as cuda
#import pycuda.cumath
#import pycuda.autoinit
#import pycuda.gpuarray as cu_array

from .parameter import Parameters
from .core.utils import ristra




################################################################################
# move ELSEWHERE
import ast, math
class IdentifierExtractor(ast.NodeVisitor):
  def __init__(self):
    self.ids = set()
  def visit_Name(self, node):
    self.ids.add(node.id)

def getStringVars(FUN):
  extractor = IdentifierExtractor()
  extractor.visit(ast.parse(FUN))
  extractor.ids = extractor.ids - set({**vars(math),**{'alpha':0}})
  return list(extractor.ids)
















def getDataFile(file_path):
  file = json.load(open(file_path))
  # New Functions
  def alpha(x, y=1):
    z = x/y
    return z*( (z.sum())/((z**2).sum()) )
  needed_vars = []
  data = uproot.open(file['path'])[file['tree_name']]
  input_vars = data.keys()
  for var in file['branches'].values():
    new_ones = getStringVars(var)
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


class Categories(object):
  """docstring for Categories."""

  def __init__(self, arg):
    super(Categories, self).__init__()
    self.arg = arg



class Sample(object):
  """
  docstring for Sample.
  """

  def __init__(self, df, name='untitled', cuts = None, params = None,
               copy=True, convert=True, trim=False, backup=False):
    self.name = name
    self.__backup = backup
    if self.__backup:
      self.__df = df                               # to maintain an orginal copy
    if cuts:
      self.df = df.query(cuts)
    else:
      self.df = df
    self.params = params

  def __get_name(self, filename):
    namewithextension = os.path.basename(os.path.normpath(filename))
    return os.path.splitext(namewithextension)[0]

  @property
  def branches(self):
    return list(self.df.keys())

  @property
  def shape(self):
    return self.df.shape()

  @classmethod
  def from_file(cls, filename, name = None, cuts = None, params = None):
    if filename[-5:] != '.json': filename += '.json'
    if not name:
      namewithextension = os.path.basename(os.path.normpath(filename))
      name = os.path.splitext(namewithextension)[0]
    return cls(getDataFile(filename), name, cuts = cuts, params = params)

  @classmethod
  def from_pandas(cls, df, name = None, cuts = None, params = None,
                  copy=True, convert=True, trim=False):
    return cls(df, name, cuts=cuts, params=params,
               copy=copy, convert=convert, trim=trim)

  def add(self, name, attribute):
    self.__setattr__(name,attribute)

  def cut(self, cuts = None):
    """
    Place cuts on df and return it with them applied!
    """
    if cuts:
      df = self.df.query(cuts)
    else:
      df = self.df
    return df

  def back_to_original(self):
    """
    If there is a backup df, load it
    """
    if self.__backup:
      self.df = self.__df

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
      #self.add(var+'_h',this_branch)
      #self.add(var+'_d',cu_array.to_gpu(this_branch).astype(np.float64))
      self.add(var, ristra.allocate(this_branch).astype(np.float64))

  def assoc_params(self, params):
    self.params = Parameters()
    self.params.copy(params)
