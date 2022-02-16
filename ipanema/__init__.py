import os
from .splot import get_exposed_package_objects
from lib99ocl import LIB99OCL

IPANEMAPATH = os.path.dirname(os.path.abspath(__file__))
IPANEMALIB = LIB99OCL


objs = get_exposed_package_objects(IPANEMAPATH)
globals().update(objs)


__all__ = ['IPANEMAPATH', 'IPANEMALIB']
__all__ += list(objs.keys())
__all__.sort()


# vim:foldmethod=marker
