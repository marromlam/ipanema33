#!/usr/bin/env python

from setuptools import setup

long_desc = """

IPANEMA: Hyperthread Curve-Fitting Module for Python

Ipanema provides a high-level interface to non-linear for Python.
It supports most of the optimization methods from scipy.optimize jointly with
others like emcc, ampgo and the so-calle Minuit.

Main functionalities:

  * Despite the comon use of plain float as fitting variables, Ipanema relies on
    the Parameter class. A Parameter has a value that can be varied in the fit,
    fixed, have upper and/or lower bounds. It can even have a value that is
    constrained by an algebraic expression of other Parameter values.

  * Multiple fitting algorithms working out-of-the-box without any change in
    the cost function to minimize.

  * Hyperthreading is avaliable and models can be compilead against different
    backends. One can use python for fits as usual, but if the amount of data
    is large, then better rewrite your code in cuda or opencl, and Ipanema can
    take care of that cost function. That's simple.

  * Estimation of confidence intervals usin ANOVA instead of calculating
    uncertainties and correlations from the covariance matrix.

Copyright (c) 2020 Ipanema Developers ; GNU AFFERO GENERAL PUBLIC LICENSE

"""


setup(name='ipanema3',
      version='1.0.1',
      author='Marcos Romero Lamas',
      author_email='mromerol@cern.ch',
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
                        'uproot3',
                        'hjson',
                        'reikna',
                        'iminuit==1.5.2',
                        'matplotlib',
                        'tqdm',
                        'corner',
                        'complot', 'lib99ocl', 
                        'pyopencl'],
      python_requires='>=3.5',
      license='GNU AFFERO GENERAL PUBLIC LICENSE',
      description="Fitting Tool for High Energy Physics",
      long_description=long_desc,
      platforms=['Linux', 'macOS', 'Windows'],
      keywords='curve-fitting, optimization, hyperthreading',
      # tests_require=['pytest'],
      # package_dir={'ipanema': 'ipanema'},
      packages=['ipanema'],
      include_package_data=True,
      )
