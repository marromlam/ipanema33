# Ipanema

<p align="center">
  <img src="ipanema_logo.png" width="250" title="Ipanema Logo">
</p>

_Ipanema_ provides a high-level interface to non-linear fitting in Python.
Developed from a High Energy Physics perspective, _Ipanema_ was mainly oriented to
fit probability density functions via maximum-likelihood fits, however it
can be used in other disciplines. Nevertheless, each problem requieres a
different approach to be minimized, sometimes gradient-based methods work like
charm, but not always. Hence _Ipanema_ supports most of the optimization methods
from _scipy.optimize_ jointly with
others bla, ble, bli and the so-called _Minuit_ developed at CERN.

Data samples are getting bigger and the bigger the data sample the longer the
fitting time. The use of accelerators like GPUs is currently among the best and
most used altenatives, thus requiring coding some parts of the fitting-model
in C/C++ to use in OpenCL and CUDA platforms. _Ipanema_ is writen on top of python
libraries making the use of OpenCL and CUDA: once the model is writen, there is no
need to worry about it, ipanema acts as an interface being those functions
user-transparent wraps.
_Ipanema_ aims to superseed packages like RooFit, making
minimization easier, handier and more user-friendly.

## Installation

This package is very easy to install. First we simply clone this repo:
```
git clone ssh://git@gitlab.cern.ch:7999/mromerol/ipanema3.git
```
and then we can install it with pip,
```
pip install -e /path/to/ipanema3/
```

### Setup with CUDA
```
function set_ipanema {
  source /path/to/virtualenv/bin/activate
  export PATH="/usr/local/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
}
```
bla bla bla
