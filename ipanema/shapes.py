# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                    DEFAULT DISTRIBUTIONS AND LINESHAPES                      #
#                                                                              #
#     Author: Marcos Romero                                                    #
#    Created: 04 - dec - 2019                                                  #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################



from numpy import arctan, cos, exp, finfo, float64, isnan, log, pi, sin, sqrt, where
from numpy.testing import assert_allclose
from scipy.special import erf, erfc
from scipy.special import gamma as gamfcn
from scipy.special import gammaln, wofz

log2 = log(2)
s2pi = sqrt(2*pi)
spi = sqrt(pi)
s2 = sqrt(2.0)
tiny = finfo(float64).eps

functions = ('gaussian_h')


def gaussian_h(x, mu=0.0, sigma=1.0, amplitude = 1):
    """
    Return a 1D Gaussian function –– evaluated at host

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))



def gaussian_d(x, mu=0.0, sigma=1.0, amplitude = 1):
    """
    Return a 1D Gaussian function –– evaluated at device     [WARNING, NOT TRUE]

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))



def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,
                         err_msg='', verbose=True):
    """Check whether all actual and desired parameter values are close."""
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol,
                        atol, err_msg, verbose)
