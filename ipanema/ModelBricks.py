# -*- coding: utf-8 -*-


#from tools import initialize
#initialize()

class Struct:
    "A structure that can have any fields defined."
    def __init__(self, **entries): self.__dict__.update(entries)



import numpy as np
import json
import pycuda.gpuarray as gpuarray
from timeit import default_timer as timer
import rotate
import  pickle as cPickle
import toygen
from pycuda.compiler import SourceModule
from scipy import random as rnd
rot = SourceModule(rotate.cu)
import os
shell = os.system
#from os import system as shell
matrixXvector = rot.get_function("transform_f32")
#from multiprocessing import Pool
#pool = Pool( processes = 24)
from iminuit import *
import pymultinest as mnest
#import genetic

import tikzplotlib

def getGrid(Nevts, BLOCK_SIZE):
    Nbunch = Nevts *1. / BLOCK_SIZE
    if Nbunch > int(Nbunch):
        Nbunch = int(Nbunch) +1
    else :
        Nbunch = int(Nbunch)
    return  (Nbunch,1,1)

def cuRead(thing, **kwargs): 
  return SourceModule(open(thing,"r").read(), **kwargs)


from prettytable import PrettyTable



# Genetic Algoritm Libraries
import random


def NumPow(X):
  Y = np.around(np.log10(abs(X)));
  Y = Y - (10 ** Y > abs(X));
  return Y

def UncRound(x, ux):
  if type(x) is float:
    x  = np.array([[x]])
    ux = np.array([[ux]])
  elif type(x) is int:
    x  = np.array([[x]])
    ux = np.array([[ux]])
  elif type(x) is np.ndarray:
    try:
      x.shape[1]
    except:
      x = x[:, np.newaxis]
      ux = ux[:, np.newaxis]
  n = NumPow(ux)
  Y = np.concatenate((x / (10 ** n), ux / (10 ** n)), axis=1)
  Y = np.concatenate((n, np.around(10 * Y) / 10), axis=1)
  # Correction if exact decimal in round.
  f, c = Y.shape
  for l in range(0, f):
    if Y[l][2] == 10:
      naux = n[l] + 1; xaux = x[l][0]; uxaux = ux[l][0]
      yaux = np.array([xaux, uxaux])
      Y[l] = np.concatenate((naux, np.around(10*yaux)/10), axis=0)
    return Y

def ErrRound(x,ux):
  X  = UncRound(x, ux)
  Y  = np.array([ X[:,1]*10**(X[:,0]) , X[:,2]*10**(X[:,0]) ])
  return Y.T



################################################################################
class Parameter: ###############################################################
  def __init__(self, name, value = 0, limits = (), # ---------------------------
                     stepsize = 0, constant = True, latex = False,
                     dtype = np.float64, blind_offset = 0., blind_sc = 1):
    self.name     = name
    self.dtype    = dtype
    self.SetValue(value)
    self.limits   = limits
    self.constant = constant
    if latex:
      self.latex    = latex
    else:
      self.latex    = name
    self.blind_offset0 = blind_offset
    self.blind_sc0     = blind_sc
    if limits:
      self.AutoStepSize()
    else:
      self.stepsize = stepsize
    if stepsize:
      self.stepsize = stepsize

  def getName(par): return par.name

  def SetValue(self, value): # -------------------------------------------------
    self.Default  = self.dtype(value)
    self.FitInit  = self.dtype(value)
    self.Value    = self.dtype(value)

  def BlindOffset(self): # -----------------------------------------------------
    return (not self.constant)*self.blind_offset0

  def BlindScale(self): # ------------------------------------------------------
    return (not self.constant)*self.blind_sc0

  def AutoStepSize(self): # ----------------------------------------------------
    self.stepsize = 0.1*abs(self.limits[1]-self.limits[0])

  def SetLimits(self, lower, upper, constant = False): # -----------------------
    if lower > upper:
      print("WARNING: Upper bound lower than lower bound for " +self.name +
            ". Reverting them to continue.")
      m_ = M*1.; m = m_; M_ = m*1.; M = M_
    self.limits   = (m,M)
    self.constant = constant
    if m > self.FitInit or M < self.FitInit:
      print("WARNING: Init value of" + self.name +
            "not inside boundaries, setting to " + str(0.5*(M-m)) + "." )
      self.FitInit = 0.5*(M-m)
    self.autoStepSize()

  def GetSettings(self): #------------------------------------------------------
    out = {self.name: self.FitInit}
    if self.limits: out .update ({"limit_" + self.name: self.limits})
    if self.stepsize: out.update({"error_" + self.name: self.stepsize})
    if self.constant: out .update({"fix_" + self.name: True})
    return out

################################################################################



################################################################################
class Cat: #####################################################################
  def __init__(self, name, ary= [], Probs_ary = [], getN = False, N = 0):
    self.name = name
    if N:
      self.BookProbs(N)        #self.Probs = gpuarray.to_gpu(np.float64(N*[0.]))
    if ary != []:
      self.SetData(ary, getN)

  def SetData(self, ary, getN = False): # --------------------------------------
    if isinstance(ary, np.ndarray):
      self.npdata = ary
      self.data   = gpuarray.to_gpu(ary)
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



################################################################################
class ParamBox: ################################################################

  def __init__(self, params, cats = []):
    self.params = params
    self.cats   = cats
    self.func_code = Struct(
        co_varnames = map(self.getName, self.params),
        co_argcount = len(self.params)
        )

    self.dc = {}; self.Parameters = {}

    for i in range(len(self.params)):
      self.dc[self.params[i].name] = i
      self.Parameters[self.params[i].name] = self.params[i]

  def getName(self, par): return par.name # ------------------------------------

  def freeParameter(self, pars): # ---------------------------------------------
    for par in pars:
      self.Parameters[par].constant = False

  def lockParameter(self, pars): # ---------------------------------------------
    for par in pars:
      self.Parameters[par].constant = True

  def setPars2FitResult(self, fit = None): # -----------------------------------
    if fit == None: fit = self.Fit
    for key in fit.values:
      self.Parameters[key].SetValue(fit.values[key])

  def getFreePars(self): # -----------------------------------------------------
    l = []
    for par in self.Parameters.keys():
      if not self.Parameters[par].constant: l.append(self.Parameters[par])
    return l
  def fitSummary(self): return FitSummary(self.Fit, self.Parameters)
  def saveFitSummary(self, name):
        c = self.FitSummary()
        c.save(name)

  def printParameters(self, limits=0):
      fitResults = self.fitSummary().results
      listID = fitResults["ID"]
      table = PrettyTable(["Fit Parameter", "Value ± ParabError"])
      table.align["Fit Parameter"]      = "l"
      table.align["Value ± ParabError"] = "c"
      if limits:
          table.align["LowerLimit"] = "c"
          table.align["UpperLimit"] = "c"
      for k in range(len(listID)):
        par = fitResults["Parameters"][listID[k]]; srow = []
        if not par["Fixed"]:
          this = ErrRound( par["Value"], par["Error"] ) [0]
          if par["Valid"]:
            srow.append(listID[k])
          else:
            srow.append(listID[k] + " * ")
          srow.append(str(this[0]) + " ± " + str(this[1]))
          table.add_row( srow )
      print(table)

  def run_with_vals(self):
      cube = len(self.params)*[0.]
      for i in range(len(self.params)):
          cube[i] = self.params[i].val
      return self(*(np.float64(cube)))

  def GeneticScan(self,Nrow,filename = "", NG = 100, f=0.5, cr = .9 ):
      frees = self.getFreePars()
      Ncol_reals = np.int32(len(frees))
      Nrow = np.int32(Nrow)
      F = np.float64(f)
      CR = np.float64(cr)

      crap1 = gpuarray.to_gpu(np.float64(Nrow*[Ncol_reals*[0.]]))
      toygen.rand.fill_uniform(crap1)
      Rmin, Rmax = [], []
      tupvars = ["chi2/F"]
      for free in frees:
          try:
              Rmin.append(free.limits[0])
              Rmax.append(free.limits[1])
              tupvars.append(free.name +"/F")
          except IndexError:
              print("Come on, some parameters don't have specified limits.")
              print("Doing nothing (sorry).")
              return
      Rmin = np.float64(Rmin)
      Rmax = np.float64(Rmax)
      reals0 = gpuarray.to_gpu(np.float64(Nrow*[Rmin]))
      dreals = crap1*gpuarray.to_gpu(np.float64(Nrow*[Rmax-Rmin]))
      reals = reals0 + dreals
      x_reals = 0.*reals0
      Rmin = gpuarray.to_gpu(Rmin)
      Rmax = gpuarray.to_gpu(Rmax)
      cost = np.float64(Nrow*[0.])
      x_cost = 0.*cost

      def fillChi2(vals,chi2, darwin):
          for i in xrange(Nrow):
              if not darwin[i]: continue ## not darwin --> the point didn't mutate, no need to calculate again the chi2
              for j in xrange(len(frees)):
                  frees[j].setVal(vals[i][j])
              chi2[i] = self.run_with_vals()
      self.genetic_db = reals.get()
      dumy = Nrow*[1.]
      fillChi2(self.genetic_db,cost, dumy)

      for i in xrange(NG):
          genetic.re_mutate(reals, x_reals,Rmin, Rmax, F, CR, Ncol_reals, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
          x_vals = x_reals.get()
          fillChi2(x_vals, x_cost, dumy)
          darwin = (x_cost < cost)
          darwin_gpu = gpuarray.to_gpu(np.float64(darwin))
          genetic.re_select(reals,x_reals,darwin_gpu, Ncol_reals, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
          self.genetic_db = reals.get()
          fillChi2(self.genetic_db,cost,darwin)

      self.genetic_cost = cost
      if filename:
          if "ROOTSYS" in os.environ.keys():
              print("filling ROOT file")
              from RTuple import RTuple as XTuple
          else:
              print("filling TEXT file")
              from XTuple import XTuple
          tup = XTuple(filename, tupvars)
          for i in xrange(Nrow):
              for j in xrange(len(frees)):
                  tup.fillItem(frees[j].name, self.genetic_db[i][j])
              tup.fillItem("chi2",cost[i])
              tup.fill()
          tup.close()



    #def constrain(self, name, m, s):

  def CreateFit(self, **kwargs): # ---------------------------------------------
    config = {}
    for par in self.params:
      config.update(par.GetSettings())
    config.update(kwargs)
    #for key in config:
    # print(key, config[key]
    self.Fit = Minuit(self, **config)


  def MinuitFit(self, **kwargs): # ---------------------------------------------
    config = {}
    for par in self.params:
      config.update(par.GetSettings())
    config.update(kwargs)
    #for key in config:
    # print(key, config[key]
    self.Fit = Minuit(self, **config)


  def GeneticFit(self, **kwargs): # --------------------------------------------
    config = {}
    for par in self.params:
      config.update(par.GetSettings())
    config.update(kwargs)
    #for key in config:
    # print(key, config[key]
    self.Fit = Minuit(self, **config)

  def createMultinestInputs(self):
      frees = self.getFreePars()
      self.mnest_frees = frees
      Nfrees = np.int32(len(frees))
      def prior(cube,ndim,nparams):
          for i in xrange (Nfrees):
              par = frees[i]
              cube[i] = par.limits[0] + (par.limits[1]-par.limits[0])*cube[i]
              #print("Prior param ",i, par.name, cube[i]
      self.mnest_prior = prior
      self.hypercube = np.float64(Nfrees*[0.])
      def mLL(cube,ndim,nparams,lnew):
          #print("crap:", ndim, nparams, lnew
          #for i in xrange(Nfrees): self.hypercube[i] = cube[i]
          for i in xrange(Nfrees): self.mnest_frees[i].setVal(cube[i])
          return -0.5*self.run_with_vals()
      self.mnest_LL = mLL

  def createMultinest(self, savedir,reset = False, **kwargs):
      self.createMultinestInputs()
      shell ("mkdir " +savedir)
      if reset: shell("rm " + savedir + "/*")
      npar = len(self.mnest_frees)
      mnest.run(self.mnest_LL, self.mnest_prior,npar, outputfiles_basename= savedir + "/1-", **kwargs)
      self.readMultinest(savedir)

  def readMultinest(self, savedir):
      self.mnest_ana =  mnest.analyse.Analyzer(len(self.mnest_frees), outputfiles_basename= savedir + "/1-")
      def sigmas():
          a = self.mnest_ana.get_mode_stats()
          v = a[u'modes'][0][u'mean']
          s = a[u'modes'][0][u'sigma']
          dc = {}
          for i in xrange(len(self.mnest_frees)):
              par = self.mnest_frees[i]
              dc[par.name] = [v[i],s[i]]
              print(par.name, v[i],"\\pm", s[i])
          return dc
      self.mnest_vals = sigmas
      self.margplot = mnest.PlotMarginalModes(self.mnest_ana)
      def plot_marginal(*args):
          n = []
          for st in args: n.append(self.dc[st])
          self.margplot.plot_marginal(*n)
      self.plot_marginal = plot_marginal




  def profileParameter(self, par,range=0,bins=20,plot=0,submin=False):
      error = self.Fit.get_param_states()[self.dc[par]].error
      value = self.Fit.get_param_states()[self.dc[par]].value
      if not range:
        range = self.Parameters[par].limits
      range = self.Parameters[par].limits
      print(par)
      x,y = self.Fit.draw_profile(par, bins=bins, bound=range, subtract_min=submin,band=True)
      if plot == 1:
        wolf = {"x":np.array(x).tolist(),"y":np.array(y).tolist(),"name":par,
                "latex":self.Parameters[par].latex,
                "value":value,
                "error":error}
        with open("./scans/"+ par + ".json", 'w') as outfile:
          json.dump(wolf, outfile)
      tikzplotlib.save("./scans/"+ par + ".tex")
      #plt.close()
      return np.array(x),np.array(y)

  def profileAll(self, range=0,bins=50,plot=1,submin=True):
      list = self.Fit.list_of_vary_param()
      import matplotlib.pyplot as plt
      for par in list:
        self.profileParameter(par,range=range,bins=bins,plot=plot,submin=submin)
        plt.show()
        plt.close()


  def ipaProfile(self, name, x0 = "", x1 = "", bins = 100, offset = True):
      setcte = 0

      if not self.Params[name]. constant :
          self.Params[name].constant = True
          setcte = 1
      x,y = [], []
      if x0 == "":
          try:
              x0 = self.Params[name].limits[0]
          except IndexError:
              print("Minimum of scan range not set, and the parameter doesn't have specified limits, so I don't know which range to scan.")
              print("Doing nothing (sorry)")
              return
      if x1 == "":
          try:
              x1 = self.Params[name].limits[1]
          except IndexError:
              print("Maximum of scan range not set, and the parameter doesn't have specified limits, so I don't know which range to scan.")
              print("Doing nothing (sorry)")
              return

      par_range = x1-x0
      for j in range(bins):
          x.append(x0 + j*par_range*1./bins)
          self.Params[name].setVal( x[j])

          self.createFit()
          self.Fit.migrad()
          y.append(self.Fit.get_fmin()['fval'])
      if setcte: self.Params[name].constant = False
      x = np.float64(x)
      y = np.float64(y)
      if offset: y -= min(y)
      #plt.plot(x,y)
      #plt.show()
      return x, y#, plt
  ## def camaron(self, par):
  ##     working = 1
  ##     while working:
  ##         try:
  ##             shit = self.Fit.minos(par, maxcall = 10000)
  ##             if shit[free.name]['is_valid']: working = 0
  ##             else :
  ##                 self.randomizeFrees()
  ##                 self.createFit()
  ##                 self.Fit.migrad()
  ##                 self.Fit.hesse()
  ##         except:
  ##             print("Algo peto. Trying another set"
  ##             self.randomizeFrees()
  ##             self.createFit()
  ##             self.Fit.migrad()
  ##             self.Fit.hesse()

  def ipa2DProfile(self, nameX,nameY, xlim = "", ylim = "", binsX = 20, binsY = 20, offset = True):
      setcteX, setcteY = 0, 0

      if not self.Params[nameX]. constant :
          self.Params[nameX].constant = True
          setcteX = 1
      if not self.Params[nameY]. constant :
          self.Params[nameY].constant = True
          setcteY = 1
      x,y = [], []#, []
      if xlim == "":
          try:
              xlim = [self.Params[nameX].limits[0], self.Params[nameX].limits[1]]
          except IndexError:
              print("X scan range not set, and the parameter doesn't have specified limits, so I don't know which range to scan.")
              print("Doing nothing (sorry)")
              return
      if ylim == "":
          try:
              ylim = [self.Params[nameY].limits[0], self.Params[nameY].limits[1]]
          except IndexError:
              print("Y scan range not set, and the parameter doesn't have specified limits, so I don't know which range to scan.")
              print("Doing nothing (sorry)")
              return
      x0 = xlim[0]
      x1 = xlim[1]
      y0 = ylim[0]
      y1 = ylim[1]
      x = np.arange(x0,x1, (x1-x0)*1./binsX)
      y = np.arange(y0,y1, (y1-y0)*1./binsY)

      X,Y = np.meshgrid(x,y)
      Z = X*0
      for j in range(binsX):
          self.Params[nameX].setVal(x[j])
          for i in range(binsY):
              self.Params[nameY].setVal(y[i])

              self.createFit()
              self.Fit.migrad()
              Z[i][j] = self.Fit.get_fmin()['fval']
      if setcteX: self.Params[nameX].constant = False
      if setcteY: self.Params[nameX].constant = False
      if offset: Z-=np.min(Z)
      #from tools import contour
      return X, Y, Z#, contour(X,Y,Z)
  def setPars2bfp(self):
      if not "fit" in dir(self):
          print("boy, I can't do that. The fit hasn't even been created yet. Come on. ")
      for key in self.Params.keys(): self.Params[key].setVal(self.Fit.values[key])
  def setPars2fitSummary(self,fs):
      for key in self.Params.keys(): self.Params[key].setVal(fs.values[key])
  def randomizeFrees(self):
      frees = self.getFreePars()
      for fr in frees: fr.setVal(fr.limits[0] + (fr.limits[1]-fr.limits[0])*rnd.random())
  def migros(self, pars = []):
      if not "fit" in dir(self):
          print("boy, I can't do that. The fit hasn't even been created yet. Come on. ")
          return
      print("****************************************************")
      print("*                      MIGROS                      *")
      print("****************************************************")
      MAXCALL = 10000
      if not pars: frees = self.getFreePars()
      else : frees = pars
      changes = False
      strategy = False
      shits = []

      for free in frees:
          working = 1
          attemps = 0
          while working:
              #if strategy: self.Fit.set_strategy(1)
              try: shit = self.Fit.minos(var = free.name, maxcall = MAXCALL).copy()
              except:
                  print("MIGROS MESSAGE: Hay cosas que petan. Randomizando a ver que pasa")
                  self.randomizeFrees()
                  self.Fit.migrad()
                  self.Fit.hesse()
                  continue
              if shit[free.name]['is_valid']: working = 0
              elif shit[free.name]['at_upper_max_fcn'] or shit[free.name]['at_lower_max_fcn'] :
                  print("Minos has an issue with MaxCall. Insisting with a larger maxcall")
                  MAXCALL*10
              elif shit[free.name]['at_upper_limit'] or shit[free.name]['at_lower_limit']:
                  print("Minos fails because ", free.name, " hits the limit. I don't dare to change them.")
                  print("Stoping")
                  return
              elif attemps <3:
                  print("Minos fails w/o saying why. It doesn't say anything about having found a better best minimum,")
                  print("yet I suspect that is what happens. Invoking migrad, and adding warning printout")
                  changes = True
                  self.Fit.migrad()
                  attemps += 1
              elif attemps<10:
                  for fr in frees: fr.setVal(fr.limits[0] + (fr.limits[1]-fr.limits[0])*rnd.random())
                  tol = self.Fit.tol
                  self.createFit()
                  self.Fit.tol = tol
                  self.Fit.migrad()
                  try: self.Fit.hesse()
                  except RuntimeError:
                      self.Fit.migrad()
                      try: self.Fit.hesse()
                      except RuntimeError:
                          self.Fit.migrad()
                          try: self.Fit.hesse()
                          except RuntimeError: print("Hesse failed. Randomizing")
              else:
                  print("giving up ", free.name)
                  working = 0
          shit["name"] = free.name
          shits.append(shit)
      print("****************************************************")
      print("*                      MIGROS                      *")
      print("****************************************************")
      for shit in shits:
          name = shit["name"]
          print(name, shit[name] ['min'], shit[name]['lower'], shit[name]['upper'], "Valid:",shit[name]['is_valid'])
          self.Params[name].setVal(shit[name]['min'])
      if changes: print("Warning, the minimum may have changed a bit")
      print("Best chi2:", self.Fit.get_fmin()['fval'])
      print("Warning, minuit strategy was loosened at least once")
      print("chi2 at the table point:", self.run_with_vals())
      return shits

class Free(Parameter): # -------------------------------------------------------
    def __init__(self, name, value = 0, limits = (),
                       stepsize = 0, latex=False, dtype = np.float64,
                       blind_offset = 0, blind_sc = 1):
      Parameter.__init__(self, name, value = value, limits = limits,
                               stepsize = stepsize,
                               constant = False,
                               latex = latex,
                               dtype = dtype,
                               blind_offset = blind_offset, blind_sc = blind_sc)
#####
############
####################

class FitSummary:
  def __init__(self, fit, Parameters):
    results = {}
    results["ErrorDef"] = fit.errordef
    results["EDM"]      = fit.edm
    results["FCN"]     = fit.get_fmin()['fval']

    results["Parameters"] = {}
    results["ID"]         = []
    __merrors = fit.get_merrors()
    __params  = fit.get_param_states()

    for par in __params:
      results["Parameters"][par.name] = {}
      results["ID"].append(str(par.name))
      results["Parameters"][par.name]["Value"]  = par.value
      results["Parameters"][par.name]["Error"] = par.error
      try:
        results["Parameters"][par.name]["LowUnc"] = __merrors[par.name]['lower']
        results["Parameters"][par.name]["UppUnc"] = __merrors[par.name]['upper']
      except:
        results["Parameters"][par.name]["LowUnc"] = par.error
        results["Parameters"][par.name]["UppUnc"] = par.error        
      if par.has_limits:
        results["Parameters"][par.name]["LL"] = par.lower_limit
        results["Parameters"][par.name]["UL"] = par.upper_limit
        check1 = (par.value>par.lower_limit) & (par.value<par.upper_limit)
        results["Parameters"][par.name]["Valid"] = check1
      else:
        results["Parameters"][par.name]["LL"] = 0.9*par.value
        results["Parameters"][par.name]["UL"] = 1.1*par.value
        results["Parameters"][par.name]["Valid"] = True
      results["Parameters"][par.name]["Fixed"] = par.is_fixed
      results["Parameters"][par.name]["LaTeX"] = Parameters[par.name].latex

    self.results = results



    self.values = fit.values
    self.errors = fit.errors
    self.cov    = np.matrix(fit.matrix())
    self.invcov = np.matrix(np.linalg.inv(self.cov))

    #self.var2pos = fit.var2pos
    #self.pos2var = fit.pos2var

    self.free  = fit.list_of_vary_param()
    self.fixed = fit.list_of_fixed_param()

    self.func_code = Struct(
        co_varnames = self.free,
        co_argcount = len(self.free)
        )
    self.table = {}
    for i in range(len(self.free)): self.table[self.free[i]] = i
    eL, eV = np.linalg.eig(self.invcov)
    self.R = np.matrix(eV)
    self.Ri = np.matrix(np.linalg.inv(self.R))
    self.eL = np.matrix(np.diag(eL))
    self.S = np.sqrt(self.eL)
    self.Si = np.linalg.inv(self.S)
    self.T = self.R*self.Si
    self.mean = np.matrix(map(self.values.get, self.free))
    self.T_gpu = gpuarray.to_gpu(np.float32(self.T))
    self.nvars = np.int32(len(self.free))

  def getResults():
      return self.R
  def save(self, fname): cPickle.dump(self,file(fname, "w"))
  #def mu(self, var): return self.values[var]
  def sigma(self,var): return self.errors[var]

  def chi2(self,point):
      Y = np.matrix(map(point.get, self.free))
      d = Y - self.mean
      print(d*self.invcov*d.transpose())

  def __call__(self, *args):
      Y = np.matrix(args)
      d = Y - self.mean
      return (d*self.invcov*d.transpose())[0][0]

  def createFit(self): self.Fit = Minuit(self)

  def rotate(self, ary):
      return self.T*np.matrix(ary).transpose()

  def generate(self, N, dtype = 'float32'):
      N2 = int(N*self.nvars)
      l = np.zeros( N2, dtype)
      ary = gpuarray.to_gpu(l)
      self.deltas_gpu = ary.copy()
      toygen.rand.fill_normal(ary)

      matrixXvector(ary,self.deltas_gpu,self.T_gpu,self.nvars, block=(1,1,1), grid = (int(N),1,1))
      self.deltas = self.deltas_gpu.get()

  def pickGenerated(self, i):
      dc = {}
      N = self.nvars
      i0 = i*N
      for i in xrange(N):
          key = self.free[i]
          dc[key] = self.deltas[i0+i] + self.values[key]
      return dc
