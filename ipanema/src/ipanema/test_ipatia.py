import ipanema
import numpy as np
ipanema.initialize('opencl',1)
from scipy import special
import matplotlib.pyplot as plt




################################################################################
################################################################################

"""
import ROOT
import root_numpy

def roofit_ipatia(x, mu, sigma, lambd, zeta, beta, aL, nL, aR, nR):
    _x = np.copy(x)
    _x.dtype = [('xx', np.float64)]
    _tree = root_numpy.array2tree(_x, "xx")
    #merda = ROOT.RooArgSet(_x)
    xx = ROOT.RooRealVar("xx", "xx", np.min(x), np.max(x))
    X = ROOT.RooDataSet("data", "data", _tree, ROOT.RooArgSet(xx))
    #X = ROOT.RooDataSet("data", "data", merda, ROOT.RooFit.Import(_tree))
    print(xx.getVal(), X)
    _mu = ROOT.RooRealVar('mu', 'mu', mu)
    _sigma = ROOT.RooRealVar('sigma', 'sigma', sigma)
    _lambd = ROOT.RooRealVar('lambd', 'lambd', lambd)
    _zeta = ROOT.RooRealVar('zeta', 'zeta', zeta)
    _beta = ROOT.RooRealVar('beta', 'beta', beta)
    _aL = ROOT.RooRealVar('aL', 'aL', aL)
    _nL = ROOT.RooRealVar('nL', 'nL', nL)
    _aR = ROOT.RooRealVar('aR', 'aR', aR)
    _nR = ROOT.RooRealVar('nR', 'nR', nR)
    #return ROOT.RooHypatia2("pdf", "pdf", X, _mu, _sigma, _lambd, _zeta, _beta, _aL, _nL, _aR, _nR)
    return ROOT.RooGaussian("pdf", "pdf", xx, _mu, _sigma)



from pyroofit.models import PDF



class Hypatia2(PDF):
    def __init__(self,
                 observable,
                 mu=(-1000, 0, 1000),
                 sigma=(0., 1, 10),
                 lambd=(-5, -1.,  5),
                 zeta=(0., 0.2, 50000),
                 beta=(-5., 0.5, 5),
                 aL=(0, 50.0, 100.0),
                 aR=(0, 1.5, 100.0),
                 nL=(0, 1.0, 100.0),
                 nR=(0, 0.1, 100.0),
                 name="hypatia", **kwds):

        super(Hypatia2, self).__init__(name=name, **kwds)

        x = self.add_observable(observable)

        _mu = self.add_parameter(mu, 'mu')
        _sigma = self.add_parameter(sigma, 'sigma')
        _lambd = self.add_parameter(lambd, 'lambd')
        _zeta = self.add_parameter(zeta, 'zeta')
        _beta = self.add_parameter(beta, 'beta')
        _aL = self.add_parameter(aL, 'aL')
        _nL = self.add_parameter(nL, 'nL')
        _aR = self.add_parameter(aR, 'aR')
        _nR = self.add_parameter(nR, 'nR')

        self.roo_pdf = ROOT.RooHypatia2("pdf", "pdf", x, _lambd, _zeta, _beta, _sigma, _mu, _aL, _nL, _aR, _nR)

def roofit_ipatia(x, mu, sigma, lambd, zeta, beta, aL, nL, aR, nR):
    pdf = Hypatia2(('x', np.min(x), np.max(x)),
                     mu=(-1000, mu, 1000),
                     sigma=(0.001, sigma, 10),
                     lambd=(-5, lambd,  5),
                     zeta=(0., zeta, 50000),
                     beta=(-5., beta, 5),
                     aL=(0, aL, 100.0),
                     nL=(0, nL, 100.0),
                     aR=(0, aR, 100.0),
                     nR=(0, nR, 100.0))
    h = pdf.roo_pdf.createHistogram( pdf.get_observable().GetName(), len(x))
    hy, hx = root_numpy.hist2array(h, False, False, True)
    hx = 0.5*(hx[0][:-1] + hx[0][1:])
    return hx, hy

"""






#pdf = Hypatia2(('x', -10, 10))
#print( roofit_ipatia(x, 0., 2., -4., 0.01, -0.0, 50., 4., 1.5, 0.1) )
#X, Y = roofit_ipatia(x, 5500., 17., -1., 0.0, -0.0, 5., 4., 1.5, 0.1)
#Y = Y/np.trapz(Y,X)
#plt.plot(x,Y)
#plt.show()
#exit()

#%% ----------------------------------------------------------------------------


prog = THREAD.compile(
"""
#define USE_DOUBLE 0
#include <ipanema/core.c>
#include <ipanema/complex.c>
#include <ipanema/special.c>
#include <ipanema/lineshapes.c>
#include <exposed/kernels.ocl>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])



def ipatia(x, mu, sigma=1, lambd=-1, zeta=0, beta=-0.01, aL=50, nL=1.5, aR=1., nR=0.1):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(0*x)
  prog.py_ipatia(yd, xd, np.float64(mu), np.float64(sigma),
                  np.float64(lambd), np.float64(zeta), np.float64(beta),
                  np.float64(aL), np.float64(nL),
                  np.float64(aR), np.float64(nR),
                  global_size=(len(x)))
  return ipanema.ristra.get(yd)






#%% ----------------------------------------------------------------------------
plt.show()
exit()

ipatia(x, 0., 2., -4., 0.01, -0.1, 50., 4., 1.5, 0.1)

def besselk(x, n):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(0*x)
  prog.py_rkv(yd, xd, np.float64(n), global_size=(len(x)))
  return ipanema.ristra.get(yd)

x = np.linspace(1.,20,20).astype(np.float64)
besselk(x, 0.5)
np.abs(besselk(x, -4.6) - special.kv(4.6, x)) / special.kv(0.6, x)

special.kv(-0.6, x)
special.kv(+0.6, x)

#%% ----------------------------------------------------------------------------



################################################################################
################################################################################



prog = THREAD.compile(
"""
#define USE_DOUBLE 1
#include<ipanema/core.cpp>
#include<ipanema/complex.cpp>
#include<ipanema/special.cpp>
#include<ipanema/lineshapes.cpp>
#include<ipanema/psIpatia.cu>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])






def ipatia(x, mu, sigma, lambd, zeta, beta, aL, nL, aR, nR):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(0*x)
  prog.py_ipatia(yd, xd, np.float64(mu), np.float64(sigma),
                  np.float64(lambd), np.float64(zeta), np.float64(beta),
                  np.float64(aL), np.float64(nL),
                  np.float64(aR), np.float64(nR),
                  global_size=(len(x)))
  return ipanema.ristra.get(yd)



def ipatia2(x, mu, sigma, lambd, zeta, beta, aL, nL, aR, nR):
  d = x-mu
  print(d)
  print(sigma, lambd, zeta, beta, aL, nL, aR, nR, -2*lambd)
  cons1 = -2*lambd;
  delta = sigma*np.sqrt(-2+cons1) if (lambd<=-1.0) else sigma
  delta2 = delta*delta
  print(delta)
  return np.exp( beta*d + (lambd-0.5)*np.log(1. + d*d/delta2) )



def hyperbolic_distribution(x, mu, lambd, alpha, beta, delta):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(0*x)
  prog.py_hyperbolic_distribution(yd, xd, np.float64(mu),
                  np.float64(lambd), np.float64(alpha), np.float64(beta),
                  np.float64(delta), global_size=(len(x)))
  return ipanema.ristra.get(yd)



def besselk(x, n):
  xd = ipanema.ristra.allocate(x)
  yd = ipanema.ristra.allocate(0*x)
  prog.py_rkv(yd, xd, np.float64(n), global_size=(len(x)))
  return ipanema.ristra.get(yd)


x = np.linspace(5000,6000,200).astype(np.float64)

ipatia(x, 0., 1., -1., 0.0, -0.01, 0., 0., 0., 0.)
plt.semilogy(x, ipatia2(x, 0., 2., -2., 0.0, -0.0, 50., 4., 1., 0.1))
plt.plot(x, ipatia(x, 0., 1., -1., 0.0, -0.01, 50., 4., 1.5, 0.1))
plt.plot(x, ipatia(x, 0., 2., -4., 0.01, -0.0, 50., 4., 1.5, 0.1))

plt.plot(x, hyperbolic_distribution(x, 0., 2., 1., 2, 2.4))
np.abs(besselk(x, 12) - special.kv(12, x)) / special.kv(12, x)

besselk(x, 0)
special.kv(0, x)






dir(roofit_ipatia(x, 0., 2., -4., 0.01, -0.0, 50., 4., 1.5, 0.1))


#%% ............................................................................
#%% ............................................................................




















prog = THREAD.compile(
"""
#define USE_DOUBLE 1

#include <ipanema/core.c>
#include <ipanema/complex.c>
#include <ipanema/special.c>


KERNEL
void pywofz2(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = cwofz(z[idx]);
}

KERNEL
void pywofz(GLOBAL_MEM const ctype *z, GLOBAL_MEM ctype *out)
{
   const int idx = get_global_id(0);
   out[idx] = cerf(z[idx]);
}



""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])


import numpy as np
from scipy.special import wofz, erfc, erf


def my_wofz(z):
  if isinstance(z, np.ndarray):
    z_dev = THREAD.to_device(np.complex128(z))
    deallocate = True
  else:
    z_dev = z
    deallocate = False
  w_dev = THREAD.to_device(np.complex128(0*z))
  prog.pywofz(z_dev, w_dev, global_size=(len(z),))
  return ipanema.ristra.get(w_dev) if deallocate else w_dev


def my_wof2(z):
  if isinstance(z, np.ndarray):
    z_dev = THREAD.to_device(np.complex128(z))
    deallocate = True
  else:
    z_dev = z
    deallocate = False
  w_dev = THREAD.to_device(np.complex128(0*z))
  prog.pywofz2(z_dev, w_dev, global_size=(len(z),))
  return ipanema.ristra.get(w_dev) if deallocate else w_dev



x = np.linspace(-4,4,1000)
y = np.linspace(-4,4,1000)
X,Y = np.meshgrid(x,y)
Z = np.stack((X.ravel(), Y.ravel()), axis=-1)
z = Z[:,0] + 1j*Z[:,1]

%time my_wofz(z)
%time my_wof2(z)

erf(z)
my_wofz(z)

np.max(np.nan_to_num(np.abs( my_wofz(z) - erf(z) )))
np.max(np.nan_to_num(np.abs( my_wof2(z) - wofz(z) )/wofz(z)))
