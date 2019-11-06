#from tools import initialize
#initialize()
import numpy as np
import pycuda.gpuarray as gpuarray
from timeit import default_timer as timer
import rotate
import cPickle
import toygen
from pycuda.compiler import SourceModule
rot = SourceModule(rotate.cu)
from os import system as shell
matrixXvector = rot.get_function("transform_f32")
#from multiprocessing import Pool
#pool = Pool( processes = 24)
from iminuit import *
import pymultinest as mnest
import genetic 




def cuRead(thing, **kwargs): return SourceModule(file(thing,"r").read(), **kwargs)
def getName(par): return par.name

################################################################################
class Parameter: ###############################################################
  def __init__(self, name, value = 0, limits = (), # ---------------------------
                     stepsize = 0, constant = True, 
                     dtype = np.float64, blind_offset = 0., blind_sc = 1):
    self.name     = name
    self.dtype    = dtype
    self.SetValue(value)
    self.limits   = limits
    self.constant = constant
    self.blind_offset0 = blind_offset
    self.blind_sc0     = blind_sc
    if limits: 
      self.AutoStepSize()
    else: 
      self.stepsize = stepsize
    if stepsize: 
      self.stepsize = stepsize
    print "ola"

  def SetValue(self, var): # ---------------------------------------------------
    self.default  = self.dtype(var)
    self.fit_init = self.dtype(var)
    self.val      = self.dtype(var)

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
    if m > self.fit_init or M < self.fit_init:
      print("WARNING: Init value of" + self.name + 
            "not inside boundaries, setting to " str(0.5*(M-m)) + "." )
      self.fit_init = 0.5*(M-m)
    self.autoStepSize()
      
  def GetSettings(self): #------------------------------------------------------
    out = {self.name: self.fit_init}
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
      self.bookProbs(N)        #self.Probs = gpuarray.to_gpu(np.float64(N*[0.]))
    if ary != []: 
      self.setData(ary, getN)
            
    def SetData(self, ary, getN = False):
      if isinstance(ary, np.ndarray):
        self.np_data = ary
        self.data = gpuarray.to_gpu(ary)
        if getN: self.bookProbs(len(ary))
      elif isinstance(ary, gpuarray.GPUArray):
        if getN:
            print "Warning: Number of events set to GPUArray size. This may be a bad idea if you are not in 1D"
            self.bookProbs(ary.size)
        self.data = ary.copy()
      elif isinstance(ary, list):
        ar = np.float64(ary)
        self.setData(ar, getN)
      elif isinstance(ary, file):
        ar = cPickle.load(ary)
        self.setData(ar, getN)
      elif isinstance(ary, str):
        f = file(ary)
        self.setData(f, getN)

        else: "dunno"
    def bookProbs(self,N):
        self.Nevts = np.int32(N)
        self.Probs = gpuarray.to_gpu(np.float64(N*[0.]))

################################################################################      

class ParamBox:
    def __init__(self, params, cats = []):
        self.params = params
        self.cats = cats
        self.func_code = Struct(
            co_varnames = map(getName, self.params),
            co_argcount = len(self.params)
            )
        self.dc = {}
        self.Params = {}
        for i in range(len(self.params)):
            self.dc[self.params[i].name] = i
            self.Params[self.params[i].name] = self.params[i]

    def freeThese(self, pars):
        for par in pars: self.Params[par].constant = False
    def lock_to_init(self, pars):
        for par in pars: self.Params[par].constant = True
    def getFreePars(self):
        l = []
        for par in self.Params.keys():
            if not self.Params[par].constant: l.append(self.Params[par])
        return l

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
            Rmin.append(free.limits[0])
            Rmax.append(free.limits[1])
            tupvars.append(free.name +"/F")
        Rmin = np.float64(Rmin)
        Rmax = np.float64(Rmax)
        reals0 = gpuarray.to_gpu(Nrow*[Rmin])
        dreals = crap1*gpuarray.to_gpu(Nrow*[Rmax-Rmin])
        reals = reals0 + dreals
        x_reals = 0.*reals0
        Rmin = gpuarray.to_gpu(Rmin)
        Rmax = gpuarray.to_gpu(Rmax)
        cost = np.float64(Nrow*[0.])
        x_cost = 0.*cost
        def fillChi2(vals,chi2):
            for i in xrange(Nrow):
                for j in xrange(len(frees)):
                    frees[j].setVal(vals[i][j])
                chi2[i] = self.run_with_vals()
        self.genetic_db = reals.get()
        fillChi2(sefl.genetic_db,cost)        
        def do():
            re_mutate(reals, x_reals,Rmin, Rmax, F, CR, Ncol_reals, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
            x_vals = x_reals.get()
            fillChi2(x_vals, x_cost)
            darwin = gpuarray.to_gpu(x_cost < cost)
            re_select(reals,x_reals,darwin, Ncol_reals, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
            self.genetic_db = reals.get()
            fillChi2(self.genetic_db,cost)
        for i in xrange(NG): do()
        self.genetic_cost = cost
        if filename:
            from XTuple import XTuple
            tup = XTuple(filename, tupvars)
            for i in xrange(Nrow):
                for j in xrange(len(frees)):
                    tup.fillItem(frees[j].name, self.genetic_db[i][j])
                tup.fillItem("chi2",cost[i])
                tup.fill()
            tup.close()
              
            
        
    #def constrain(self, name, m, s):
    def createFit(self, **kwargs):
        config = {}
        for par in self.params: config.update(par.getSettings())
        config.update(kwargs)
        self.fit = Minuit(self, **config)
    def createMultinestInputs(self):
        def prior(cube,ndim,nparams):
            for i in xrange (len(self.params)):
                par = self.params[i]
                if par.constant: cube[i] = par.val
                else: cube[i] = par.limits[0] + (par.limits[1]-par.limits[0])*cube[i]
                #print "Prior param ",i, par.name, cube[i]
        self.mnest_prior = prior
        self.hypercube = np.float64(len(self.params)*[0.])
        def mLL(cube,ndim,nparams,lnew):
            #print "crap:", ndim, nparams, lnew
            for i in xrange(len(self.params)): self.hypercube[i] = cube[i]
            return -0.5*self(*(self.hypercube))
        self.mnest_LL = mLL
    def createMultinest(self, savedir,reset = False, **kwargs):
        self.createMultinestInputs()
        shell ("mkdir " +savedir)
        if reset: shell("rm " + savedir + "/*")
        npar = len(self.params)
        mnest.run(self.mnest_LL, self.mnest_prior,npar, outputfiles_basename= savedir + "/1-", **kwargs)
        self.readMultinest(savedir)
        
    def readMultinest(self, savedir):
        self.mnest_ana =  mnest.analyse.Analyzer(len(self.params), outputfiles_basename= savedir + "/1-")
        def sigmas():
            a = self.mnest_ana.get_mode_stats()
            v = a[u'modes'][0][u'mean']
            s = a[u'modes'][0][u'sigma']
            dc = {}
            for i in xrange(len(self.params)):
                par = self.params[i]
                dc[par.name] = [v[i],s[i]]
                print par.name, v[i],"\\pm", s[i]
            return dc
        self.mnest_vals = sigmas
        self.margplot = mnest.PlotMarginalModes(self.mnest_ana)
        def plot_marginal(*args):
            n = []
            for st in args: n.append(self.dc[st])
            self.margplot.plot_marginal(*n)
        self.plot_marginal = plot_marginal
        
    def fitSummary(self): return FitSummary(self.fit)
    def saveFitSummary(self, name):
        c = self.fitSummary()
        c.save(name)

    
class Free(Parameter):
    def __init__(self, name, var = 0, limits = (), stepsize = 0, dtype = np.float64, blind_offset = 0, blind_sc = 1): Parameter.__init__(self, name, var = var, limits = limits, stepsize = stepsize, constant = False, dtype = dtype, blind_offset = blind_offset, blind_sc = 1)

class FitSummary:
    def __init__(self, fit):
        self.values = fit.values
        self.errors = fit.errors
        self.C = np.matrix(fit.matrix())
        self.cinv = np.matrix(np.linalg.inv(self.C))
        
        #self.var2pos = fit.var2pos
        #self.pos2var = fit.pos2var
        self.free = fit.list_of_vary_param()
        self.func_code = Struct(
            co_varnames = self.free,
            co_argcount = len(self.free)
            )
        self.table = {}
        for i in range(len(self.free)): self.table[self.free[i]] = i
        eL, eV = np.linalg.eig(self.cinv)
        self.R = np.matrix(eV)
        self.Ri = np.matrix(np.linalg.inv(self.R))
        self.eL = np.matrix(np.diag(eL))
        self.S = np.sqrt(self.eL)
        self.Si = np.linalg.inv(self.S)
        self.T = self.R*self.Si
        self.mean = np.matrix(map(self.values.get, self.free))
        self.T_gpu = gpuarray.to_gpu(np.float32(self.T))
        self.nvars = np.int32(len(self.free))

        
    def save(self, fname): cPickle.dump(self,file(fname, "w"))
    #def mu(self, var): return self.values[var]
    def sigma(self,var): return self.errors[var]
    
    def chi2(self,point):
        Y = np.matrix(map(point.get, self.free))
        d = Y - self.mean
        print d*self.cinv*d.transpose()
    
    def __call__(self, *args):
        Y = np.matrix(args)
        d = Y - self.mean
        return (d*self.cinv*d.transpose())[0][0] 
    
    def createFit(self): self.fit = Minuit(self)
    
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
        
        
