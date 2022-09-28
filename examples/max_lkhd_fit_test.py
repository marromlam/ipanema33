#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Simple fit

# Imports ----------------------------------------------------------------------
#    First we should do some imports
from ipanema import Parameter, Parameters, optimize, ristra
import matplotlib.pyplot as plt
import numpy as np
import os
from ipanema import initialize

initialize("cuda", 1)  # python|opencl|cuda this script runs on all backends


# %% Building the dataset to fit ------------------------------------------------
#    We create a ipanema.Parameters object with the true parameters.
p_true = Parameters()
mu = Parameter("mu", value=5.0, latex=f"\mu_1")
sigma1 = Parameter("sigma1", value=2.0, latex=r"\sigma_1")
sigma2 = Parameter("sigma2", value=6.0, latex=r"\sigma_2")
p_true.add(mu, sigma1, sigma2)
print(p_true.dump_latex())


# %% Create a model -------------------------------------------------------------
#    Write down a model. Here we are writing a model that is at the same time
#    the model (it no data is provided), or the residual fcn to optimize if the
#    data is provided.


def model(x, mu=0, sigma=1):
  """
  Gaussian pdf
  """
  argx = 0.5 * ((x - mu) / sigma) ** 2  # exp argument
  norm = sigma * np.sqrt(2 * np.pi)  # gaussian norm

  return ristra.exp(-argx) / norm


def loglikelihood(pars, data1, data2):
  """
  Simultaneous 2 gaussian fcn.
  Here one should UNBLIND the parameters so the model function reads them
  properly, that's why we use pars.valuesdict(blind=False)
  """
  vals = pars.valuesdict(blind=False)  # get parameters as simple dict

  p1 = ristra.get(ristra.log(
      model(data1, mu=vals["mu"], sigma=vals["sigma1"])))
  p2 = ristra.get(ristra.log(
      model(data2, mu=vals["mu"], sigma=vals["sigma2"])))
  prob = -2 * np.concatenate((p1, p2))  # -2 x log Likelihood here!

  return prob


# Plot the model
#    Using rista.allocate we allocate array in backend memory, example if
#    backend is python, then x will be np.ndarray; if cuda, it will
#    have allocator in device...
x = ristra.allocate(np.linspace(-10, 20, 200))
y = model(x, 4, 6)
plt.plot(ristra.get(x), ristra.get(y))
plt.show()

# %% Prepare arrays -------------------------------------------------------------
#    It's time to create a random dataset that follows the model with p_true
#    set of parameters.
N = int(1e8)
data1 = np.random.normal(
    p_true["mu"].value, p_true["sigma1"].value, size=int(1e6))
data1 = ristra.allocate(data1)  # allocate it wherever the backend needs
data2 = np.random.normal(
    p_true["mu"].value, p_true["sigma2"].value, size=int(1e5))
data2 = ristra.allocate(data2)

hdata1 = histogram.hist(ristra.get(data1), weights=None, bins=100)  # histo
hdata2 = histogram.hist(ristra.get(data2), weights=None, bins=100)  # histo
plt.close()
fig, axplot = plotting.axes_plot()
axplot.fill_between(
    hdata1.bins, hdata1.counts, step="mid", color="k", alpha=0.2, label=f"dataset 1"
)
plt.close()
fig, axplot = plotting.axes_plot()
axplot.fill_between(
    hdata2.bins, hdata2.counts, step="mid", color="k", alpha=0.2, label=f"dataset 2"
)
plt.show()


# %% Fitting --------------------------------------------------------------------
#   Lets create a p_fit datset of parameters. Here we can see a way to add
#   parameters without construction a Parameter object beforehand.
p_fit = Parameters()
p_fit.add({"name": "mu", "value": 0, "min": 0, "max": 10, "latex": r"\mu"})
p_fit.add(
    {
        "name": "sigma1",
        "value": 2,
        "min": 0.1,
        "max": 10,
        "latex": r"\sigma1",
        "blind": "blindstr",
    }
)
p_fit.add({"name": "sigma2", "value": 1, "min": 0.1,
          "max": 10, "latex": r"\sigma2"})
p_fit.print()

print(f"Parameters blinded: {p_fit.valuesdict()}")
print(f"Parameters unblinded: {p_fit.valuesdict(False)}")


# Run the fit with BFGS method
result_hesse = optimize(
    fcn_call=loglikelihood,
    method="minuit",
    params=p_fit,
    fcn_kwgs={"data1": data1, "data2": data2},
    verbose=False,  # this is to print step by step iterations
)
print(result_hesse.params.valuesdict())
print(result_hesse.params.valuesdict(False))
print(result_hesse.__str__(corr=0))

print("The fit result shows (hesse)")
print(f"params = {result_hesse.params.valuesdict()}")
print("while their actual value is")
print(f"params = {result_hesse.params.valuesdict(False)}")


# Run the fit with BFGS optimization
result_bfgs = optimize(
    fcn_call=loglikelihood,
    method="bfgs",
    params=p_fit,
    fcn_kwgs={"data1": data1, "data2": data2},
    verbose=False,  # this is to print step by step iterations
)
print(result_bfgs)

print("The fit result shows (bfgs)")
print(f"params = {result_hesse.params.valuesdict()}")
print("while their actual value is")
print(f"params = {result_hesse.params.valuesdict(False)}")


# %% Plot the fit over the data
for dataset, n in zip([data1, data2], ["1", "2"]):
  plt.close()

  # linspace arrays and eval plot with each result
  x = ristra.allocate(np.linspace(
      ristra.min(dataset), ristra.max(dataset), 200))
  y_hesse = model(
      x,
      result_hesse.params["mu"]._getval(False),
      result_hesse.params[f"sigma{n}"]._getval(False),
  )
  y_bfgs = model(
      x,
      result_bfgs.params["mu"]._getval(False),
      result_bfgs.params[f"sigma{n}"]._getval(False),
  )

  # move arrays to host, since plt wants them there
  x = ristra.get(x)
  y_hesse = ristra.get(y_hesse)
  y_bfgs = ristra.get(y_bfgs)

  # histogram data
  hdata = histogram.hist(ristra.get(dataset), weights=None, bins=100)

  # scale linspace to histogram
  factor = hdata.norm  # *abs(hdata.edges[1]-hdata.edges[0])
  y_hesse *= factor / (y_hesse.sum() * abs(x[1] - x[0]))
  y_bfgs *= factor / (y_bfgs.sum() * abs(x[1] - x[0]))

  # plot
  fig, axplot, axpull = plotting.axes_plotpull()
  axplot.fill_between(
      hdata.bins, hdata.counts, step="mid", color="k", alpha=0.2, label=f"dataset {n}"
  )
  axplot.plot(x, y_hesse, label="minuit")
  axplot.plot(x, y_bfgs, label="bfgs", color="C3")
  # axplot.set_yscale('log')
  axpull.fill_between(
      hdata.bins,
      histogram.pull_pdf(
          x, y_hesse, hdata.bins, hdata.counts, hdata.errl, hdata.errh
      ),
      0,
      facecolor="C0",
      alpha=0.5,
  )
  axpull.fill_between(
      hdata.bins,
      histogram.pull_pdf(x, y_bfgs, hdata.bins,
                         hdata.counts, hdata.errl, hdata.errh),
      0,
      facecolor="C3",
      alpha=0.5,
  )
  axplot.legend()
  fig.savefig(f"dataset{n}_fit.pdf")
  fig.show()
