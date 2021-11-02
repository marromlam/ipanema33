import importlib
import inspect
import os

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

exposed_modules = ('host', 'utils')

pkg = os.path.normpath(
    PACKAGE_PATH[PACKAGE_PATH.rfind('ipanema'):]).replace(os.sep, '.')

__all__ = []
for m in exposed_modules:
    
  try:
    mod = importlib.import_module('.' + m, package=pkg)
    for n, c in inspect.getmembers(mod):
        if n in mod.__all__:
            globals()[n] = c

    __all__ += mod.__all__
  except:
    0


__all__ = list(sorted(__all__))
