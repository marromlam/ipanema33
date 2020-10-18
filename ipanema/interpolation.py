from scipy.interpolate import interp1d
from scipy import arange, array, exp


__all__ = ['extrap1d']

def extrap1d(interpolator):
  xs = interpolator.x
  ys = interpolator.y

  def pointwise(x):
    if x < xs[0]:
      return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
    elif x > xs[-1]:
      return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
    else:
      return interpolator(x)

  def ufunclike(xs):
    return array(list(map(pointwise, array(xs))))

  return ufunclike
