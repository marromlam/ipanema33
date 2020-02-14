#!/usr/bin/env python

from setuptools import setup

long_desc = """

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



setup(name='ipanema',
      version='0.2',
      author='Marcos Romero',
      author_email='marcos.romero.lamas@cern.ch',
      url='https://github.com/marromlam/ipanema.git',
      download_url='https://github.com/marromlam/ipanema.git',
      install_requires=['asteval>=0.9.12',
                        'numpy>=1.10',
                        'scipy>=0.19',
                        'six>1.10',
                        'uncertainties>=3.0',
                        'pandas',
                        'numdifftools',
                        'emcee>=3.0',
                        'uproot',
                        'hjson',
                        'reikna',
                        'iminuit',
                        'matplotlib'],
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
      license='MIT',
      description="Fitting Tool for High Energy Physics",
      long_description=long_desc,
      platforms=['Linux', 'Mac OS X', 'Windows'],
      classifiers=['Development Status :: 0 - Production/Unstable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering',
                   ],
      keywords='curve-fitting, minimization',
      #tests_require=['pytest'],
      package_dir={'ipanema': 'ipanema'},
      packages=['ipanema'],
)