"""

IPANEMA: Hyperthread Curve-Fitting Module for Python

Ipanema provides a high-level interface to non-linear fitting for Python.
It supports most of the optimization methods from scipy.optimize jointly with
others like emcc, ampgo and the so-called CERN Minuit.

Main functionalities:

  * Despite the common use of plain float as fitting variables, ipanema relies
    on the Parameter class. A Parameter has a value that can be varied in the
    fit, fixed, have upper and/or lower bounds. It can even have a value
    constrained by an algebraic expression of other Parameter values.

  * Multiple fitting algorithms working out-of-the-box without any change in
    the cost function to minimize.

  * Hyperthreading is avaliable and models can be compiled against different
    backends. One can use python for fits as usual, but if the amount of data
    is large, then better rewrite your code in cuda or opencl, and ipanema can
    take care of that cost function. That's simple.

  * Improved estimation of confidence intervals. While most methods in
    ipanema can automatically estimate uncertainties and correlations from the
    covariance matrix (that is, the hessian), ipanema also provides functions
    to explore the parameter space and calculate confidence intervals.
    [ALMOST, BUT NOT YET!]

Copyright (c) 2020 Ipanema Developers ; GNU AFFERO GENERAL PUBLIC LICENSE

"""

## WARNING: make sure the following imports make that all necessary code is
##          avaliable for the final user

from asteval import Interpreter

# Samples
from .samples import Sample, get_data_file

# Core utils
from .core.utils import initialize
from .core.utils import ristra

# Optimize
from .optimizers import Optimizer, optimize

# Parameters
from .parameter import Parameter, Parameters, isParameter

# Confidence
from .confidence import confidence_interval, confidence_interval2d

# Tools and utils
from .tools.uncertainties_wrapper import wrap_unc, get_confidence_bands
from .utils.print_reports import fit_report
from .core import utils

# Plot related stuff
from .plot import histogram
from .plot.histogram import hist
from .plot import untitled as plotting

# Useful variables
from .optimizers import ALL_METHODS
all_optimize_methods = list(ALL_METHODS.keys()); del ALL_METHODS

from .interpolation import extrap1d
