"""

IPANEMA: Hyperthread Curve-Fitting Module for Python

Ipanema provides a high-level interface to non-linear for Python.
It supports most of the optimization methods from scipy.optimize jointly with
others like emcc, ampgo and the so-calle Minuit.

Main functionalities:

  * Despite the comon use of plain float as fitting variables, Ipanema relies on
    the Parameter class.  A Parameter has a value that can be varied in the fit,
    fixed, have upper and/or lower bounds. It can even have a value that is
    constrained by an algebraic expression of other Parameter values.

  * Multiple fitting algorithms working out-of-the-box without any change in
    the cost function to minimize.

  * Hyperthreading is avaliable and models can be compilead against different
    backends. One can use python for fits as usual, but if the amount of data
    is large, then better rewrite your code in cuda or opencl, and Ipanema can
    take care of that cost function. That's simple.

  * Improved estimation of confidence intervals. While
    scipy.optimize.leastsq() will automatically calculate uncertainties
    and correlations from the covariance matrix, lmfit also has functions
    to explicitly explore parameter space to determine confidence levels
    even for the most difficult cases.

  * Improved curve-fitting with the Model class. This extends the
    capabilities of scipy.optimize.curve_fit(), allowing you to turn a
    function that models your data into a Python class that helps you
    parametrize and fit data with that model.

  * Many built-in models for common lineshapes are included and ready
    to use.

Copyright (c) 2020 Ipanema Developers ; MIT License ; see LICENSE

"""

from asteval import Interpreter

# Samples
from .samples import Sample, getDataFile

# Core utils
from .core.utils import initialize
from .core.utils import ristra
from .core import utils

from .confidence import conf_interval, conf_interval2d
from .optimizers import Optimizer, OptimizerException, optimize
from .parameter import Parameter, Parameters

from .tools.uncertainties_wrapper import wrap_unc, get_confidence_bands

# Utils
from .utils.printfuncs import (ci_report, fit_report, report_ci, report_errors,
                         report_fit)

#from .model import Model, CompositeModel
#from . import shapes, models
from .plot import histogram
from .plot.histogram import hist
from .plot import untitled as plotting
