import numpy as np
import scipy as sc
#import sympy as sp
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import iminuit
from Parameter import *


class ClassName():
  """docstring for ClassName"""
  def __init__(self, catname, catdata, get_len = False):
    super(ClassName, self).__init__()
    self.data_cpu = set_data(catdata)

  def get_cpudata():
    return self.data_cpu
    
  def set_data(self, ary, get_len = False): # ----------------------------------
    if isinstance(ary, np.ndarray):
      self.data_cpu = ary
      self.data_gpu = gpuarray.to_gpu(ary)
      if getN: self.BookProbs(len(ary))
    elif isinstance(ary, gpuarray.GPUArray):
      if get_len():
        print("WARNING: Number of events set to GPUArray size." +
              "This may be a bad idea if you are not in 1D")
    elif isinstance(ary, list): 
      ar = np.float64(ary);   self.SetData(ar, getN)
    elif isinstance(ary, str):  
      f  = cPickle.load(open(ary)); self.SetData(f, getN)
    elif isinstance(ary, file): 
      ar = cPickle.load(ary); self.SetData(ar, getN)
    else: "ERROR: No method exists for this input instance."




################################################################################
class Dataset(OrderedDict): ####################################################

  # def __init__(self, name, ary= [], Probs_ary = [], getN = False, N = 0):
  #   self.name = name
  #   if N:
  #     self.BookProbs(N)        #self.Probs = gpuarray.to_gpu(np.float64(N*[0.]))
  #   if ary != []:
  #     self.SetData(ary, getN)

  def __init__(self, asteval=None, usersyms=None, *args, **kwds):
    super(Dataset, self).__init__(self)

  def add(self, catname, dpath):
    # if isinstance(name, Parameter):
    #   self.__setitem__(name.name, name)
    # else:
    #   self.__setitem__(name, 0)




  def set_data(self, ary, getN = False): # --------------------------------------
    if isinstance(ary, np.ndarray):
      self.data_cpu = ary
      self.data_gpu = gpuarray.to_gpu(ary)
      if getN: self.BookProbs(len(ary))
    elif isinstance(ary, gpuarray.GPUArray):
      if getN:
        print("WARNING: Number of events set to GPUArray size." +
              "This may be a bad idea if you are not in 1D")
        self.BookProbs(ary.size)
      self.data = ary.copy()
    elif isinstance(ary, list): ar = np.float64(ary);   self.SetData(ar, getN)
    elif isinstance(ary, str):  f  = cPickle.load(open(ary)); self.SetData(f, getN)
    elif isinstance(ary, file): ar = cPickle.load(ary); self.SetData(ar, getN)
    else: "ERROR: No method exists for this input instance."

  def BookProbs(self,N): # -----------------------------------------------------
    self.Nevts = np.int32(N)
    self.Probs = gpuarray.to_gpu(np.float64(N*[0.]))

################################################################################