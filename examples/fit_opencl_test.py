# [ipanema example]: opencl

# Imports ----------------------------------------------------------------------
import ipanema
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer


# %% Prepare context -----------------------------------------------------------
#    This line will create a THREAD variable that handles the kernel compilers
#    and allocation of memort in backend if it is different from python.

ipanema.core.utils.fetch_devices()  # just to have a quick lookup
ipanema.initialize("opencl", 1, verbose=True)


# %% Create a set of ipanema parameters ----------------------------------------
#    bla bla bla
pars = ipanema.Parameters()
pars.add({"name": "mu", "value": 3, "latex": "\mu"})
pars.add({"name": "sigma", "value": 5, "latex": "\sigma"})


# %% Prepare kernel model ------------------------------------------------------
#    This should be writen in reikna syntax...

kernel = THREAD.compile(
    """
KERNEL
void gaussian(GLOBAL_MEM double *x, GLOBAL_MEM double *y,
              float mu,  float sigma, int N )
{
  const SIZE_T i = get_global_id(0);
  if (i < N)
  {
    y[i] = exp( -0.5 * ((x[i]-mu)*(x[i]-mu)) / (sigma*sigma) );
    y[i] *= 1/sqrt(2*3.1415*sigma*sigma);
    //printf("%lf, %lf\\n", x[i], y[i]);
  }
}"""
)


# Wrap the kerner in a python function
def model(data, lkhd, mu, sigma):
  kernel.gaussian(
      data,
      lkhd,
      np.float32(mu),
      np.float32(sigma),
      np.int32(data.shape[0]),
      local_size=256,
      global_size=int(256 * np.ceil(data.shape[0] / 256)),
  )


# Create cost function, in this example we will proccedd with a likelihod one
def likelihood(pars, x, prob=None):
  p = pars.valuesdict()
  if prob is None:
    prob = ipanema.ristra.allocate(0 * x.get())
    return model(x, prob, p["mu"], p["sigma"]).get()
  model(x, prob, p["mu"], p["sigma"])
  return -2 * ipanema.ristra.log(prob).get() + 2 * x.shape[0]


# %% Prepare arrays ------------------------------------------------------------

# Create a random variable
m, s = pars["mu"].value, pars["sigma"].value
np.random.seed(0)
x_h = np.random.normal(loc=m, scale=s, size=1000000)
pandas_host = pd.DataFrame({"x": x_h})

# Create an ipanema sample from p
sample = ipanema.Sample.from_pandas(pandas_host)

# Allocate x and prob in device
sample.allocate(x="x", prob="0*x")


# %% Fit and get the results ---------------------------------------------------

# Create an instance of ipanema.Optimizer
result = ipanema.Optimizer(
    likelihood, params=pars, fcn_args=(sample.x, sample.prob), policy="filter"
)

# Minimize likelihood using given method
# print(result.optimize(method='emcee'))  # Markov Chain MC optimizer
print(result.optimize(method="minuit"))  # Minuit optimizer (hesse)
print(result.optimize(method="bfgs"))  # scipy Broyden optimizer

# ipanema.all_optimize_methods # all ipanema optimize methods are here
