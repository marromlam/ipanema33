# test_crystall_ball
#
#


__all__ = []
__author__ = ["Marcos Romero"]
__email__ = ["mromerol@cern.ch"]


# Modules {{{

import os
import ipanema
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import complot


ipanema.initialize('opencl', 1)
prog = ipanema.compile(
"""
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
"""
)

# }}}


# mass model {{{

def mass_model(mass, prob, fsig=1, fexp=0,
               mu=0, sigma=10, aL=0, nL=0, aR=0, nR=0, b=0,
               norm=1, mLL=None, mUL=None):
  _sigma = sigma + 0 * mass
  prog.kernel_double_crystal_ball(prob, mass, np.float64(mu), _sigma,
                                  np.float64(aL), np.float64(nL),
                                  np.float64(aR), np.float64(nR),
                                  np.float64(mLL), np.float64(mUL),
                                  global_size=(len(mass)),)
  signal_pdf = ipanema.ristra.get(prob)
  prog.kernel_exponential(prob, mass, np.float64(b), np.float64(mLL),
                          np.float64(mUL), global_size=(len(mass)),)
  background_pdf = ipanema.ristra.get(prob)
  ans = fsig * signal_pdf + fexp * background_pdf
  return norm * ans

# }}}


if __name__ == '__main__':

  # mass range
  mLL = -10
  mUL = 5

  # create a parameter set
  pars = ipanema.Parameters()
  pars.add({"name": "fsig", "value": 1, "min":0, "max":1, "free":True,
            "latex": "f_{CB}"})
  pars.add({"name": "fexp", "formula": "1-fsig", "min":0, "max":1,
            "latex": "f_{Comb}"})
  pars.add({"name": "mu", "value": 1.1,
            "latex": r"\mu"})
  pars.add({"name": "sigma", "value": 0.5, "min":0, "max":1,
            "latex": r"\sigma"})
  pars.add({"name": "aL", "value": 1,
            "latex": "a_l"})
  pars.add({"name": "nL", "value": 1,
            "latex": "n_l"})
  pars.add({"name": "aR", "value": 1,
            "latex": "a_r"})
  pars.add({"name": "nR", "value": 10,
            "latex": "n_r"})
  pars.add({"name": "b", "value": 0.0, "min":-1, "max":0,
            "latex": "b"})
  pars.add({"name": "mLL", "value": mLL,
            "latex": "m_l", "free":False})
  pars.add({"name": "mUL", "value": mUL,
            "latex": "m_u", "free":False})


  # generate dataset (if it is not there) {{{

  if not os.path.exists("test_crystal_ball.root"):
    print("Generating data")
    # lets generate a random histogram
    mass_h = np.linspace(mLL, mUL, 1000)
    mass_d = ipanema.ristra.allocate(mass_h)
    prob_d = 0*mass_h

    def lambda_model(mass):
      _mass = ipanema.ristra.allocate(mass).astype(np.float64)
      _prob = 0 * _mass
      return mass_model(mass=_mass, prob=_prob, **pars.valuesdict(), norm=1)

    pdfmax = np.max(ipanema.ristra.get(lambda_model(mass_h)))
    prob_h = ipanema.ristra.get(lambda_model(mass_h))
    print("pdfmax =", pdfmax)

    def generate_dataset(n, pdfmax=1):
      i = 0
      output = np.zeros(n)
      while i < n:
          V = (mUL-mLL) * np.random.rand() + mLL
          U = np.random.rand()
          pdf_value = lambda_model(np.float64([V]))[0]
          if U < 1/pdfmax * pdf_value:
              output[i] = V
              i = i + 1
      return output

    data_h = generate_dataset(int(1e4), 1.2*pdfmax)
    pandas_host = pd.DataFrame({"mass": data_h})
    with uproot.recreate("test_crystal_ball.root") as f:
      _branches = {}
      for k, v in pandas_host.items():
          if 'int' in v.dtype.name:
              _v = np.int32
          elif 'bool' in v.dtype.name:
              _v = np.int32
          else:
              _v = np.float64
          _branches[k] = _v
      mylist = list(dict.fromkeys(_branches.values()))
      f["DecayTree"] = uproot.newtree(_branches)
      f["DecayTree"].extend(pandas_host.to_dict(orient='list'))


    hdata = complot.hist(data_h, bins=60)
    plt.plot(hdata.bins, hdata.counts)
    plt.savefig("cb_histo.png")

  # }}}


  # fit data {{{

  print("Loading sample")
  sample = ipanema.Sample.from_root("test_crystal_ball.root")
  sample.allocate(mass="mass", prob="0*mass")

  # likelihood funtiion to optimize
  def fcn(params, data):
    p = params.valuesdict()
    prob = mass_model(mass=data.mass, prob=data.prob, **p)
    return -2.0 * np.log(prob)

  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':sample}, method='minuit',
                         verbose=False)
  print(res)
  fpars = ipanema.Parameters.clone(res.params)

  # }}}


  # plot {{{

  _p = fpars.valuesdict()
  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(ipanema.ristra.get(sample.mass), bins=100, density=False)
  axplot.errorbar(
      hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
  )

  mass = ipanema.ristra.linspace(ipanema.ristra.min(sample.mass), ipanema.ristra.max(sample.mass), 200)
  signal = 0 * mass

  # plot signal: nbkg -> 0 and nexp -> 0
  for icolor, pspecie in enumerate(fpars.keys()):
    if pspecie.startswith('f'):
      _p = ipanema.Parameters.clone(fpars)
      for f in _p.keys():
            if f.startswith('f'):
              if f != pspecie:
                _p[f].set(value=0, min=-np.inf, max=np.inf)
              else:
                _p[f].set(value=fpars[pspecie].value, min=-np.inf, max=np.inf)

      _x, _y = ipanema.ristra.get(mass), ipanema.ristra.get(
          mass_model(mass, signal, **_p.valuesdict(), norm=hdata.norm)
      )
      _label = f"${fpars[pspecie].latex.split('f_')[-1]}$"
      axplot.plot(_x, _y, color=f"C{icolor+1}", label=_label)

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  x, y = ipanema.ristra.get(mass), ipanema.ristra.get(
      mass_model(mass, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  axplot.plot(x, y, color="C0")
  pulls = complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr)
  axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)

  # label and save the plot
  axpull.set_xlabel(r"$m$ [MeV/$c^2$]")
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_yticks([-5, 0, 5])
  axpull.hlines(+3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axpull.hlines(-3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axplot.set_ylabel(r"Candidates")
  axplot.legend(loc="upper right", prop={'size': 8})
  fig.savefig("test_crystal_ball_fit.pdf")
  plt.close()

  # }}}


# vim: fdm=marker
