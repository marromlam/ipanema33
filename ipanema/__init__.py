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

Copyright (c) 2020 Ipanema Developers ; GNU AFFERO GENERAL PUBLIC LICENSE

"""

from asteval import Interpreter

# Samples
from .samples import Sample, get_data_file

# Core utils
from .core.utils import initialize
from .core.utils import ristra

# Optimize
from .optimizers import Optimizer, OptimizerException, optimize

# Parameters
from .parameter import Parameter, Parameters, isParameter

# Confidence
from .confidence import conf_interval, conf_interval2d

# Tools and utils
from .tools.uncertainties_wrapper import wrap_unc, get_confidence_bands
from .utils.print_reports import fit_report
from .core import utils

# Plot related stuff
from .plot import histogram
from .plot.histogram import hist
from .plot import untitled as plotting
