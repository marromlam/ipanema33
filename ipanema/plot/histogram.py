################################################################################
#                                                                              #
#                                  HISTOGRAM                                   #
#                                                                              #
################################################################################

from scipy.stats import chi2
from scipy.optimize import fsolve
import math
from .untitled import ipo
import numpy as np
from scipy.interpolate import interp1d




def errors_poisson(data, a=0.318):
  """
  Uses chisquared info to get the poisson interval.
  """
  low, high = chi2.ppf(a/2, 2*data) / 2, chi2.ppf(1-a/2, 2*data + 2) / 2
  return np.array(data-low), np.array(high-data)



def errors_sW2(x, weights=None, range=None, bins=60):
  if weights is not None:
    values = np.histogram(x, bins, range, weights=weights*weights)[0]
  else:
    values = np.histogram(x, bins, range)[0]
  return np.sqrt(values)



def pull_hist(ref_counts, counts, counts_l, counts_h):
  """
  This function takes an array of ref_counts (reference histogram) and three
  arrays of the objective histogram: counts, counts_l (counts' lower limit) and
  counts_h (counts' higher limit). It returns the pull of counts wrt ref_counts.
  """
  residuals = counts - ref_counts;
  pulls = np.where(residuals>0, residuals/counts_l, residuals/counts_h)
  return pulls


def pull_pdf(x_pdf, y_pdf, x_hist, y_hist, y_l, y_h):
  """
  This function compares one histogram with a pdf. The pdf is given with two
  arrays x_pdf and y_pdf, these are interpolated (and extrapolated if needed),
  contructing a cubic spline. The histogram takes x_hist (bins), y_hist(counts),
  y_l (counts's lower limit) and y_h (counts' upper limit). The result is a
  pull array between the histogram and the pdf.
  (the pdf is expected to be correctly normalized)
  """
  s = interp1d(x_pdf, y_pdf, kind='cubic', fill_value='extrapolate')
  residuals = y_hist - s(x_hist);
  pulls = np.where(residuals>0, residuals/y_l, residuals/y_h)
  return pulls



def hist(data, bins=60, weights=None, density=False, **kwargs):
  """
  This function is a wrap arround np.histogram so it behaves similarly to it.
  Besides what np.histogram offers, this function computes the center-of-mass
  bins ('cmbins') and the lower and upper limits for bins and counts. The result
  is a ipo-object which has several self-explanatory attributes.
  """

  # Histogram data
  counts, edges = np.histogram(data, bins=bins, weights=weights, density=False,
                               **kwargs)
  bincs = (edges[1:]+edges[:-1])*0.5;
  norm = counts.sum()

  # Compute the mass-center of each bin
  cmbins = np.copy(bincs)
  for k in range(0,len(edges)-1):
    if counts[k] != 0:
      cmbins[k] = np.median( data[(data>=edges[k]) & (data<=edges[k+1])] )

  # Compute the error-bars
  if weights is not None:
    errl, errh = errors_poisson(counts)
    errl = errl**2 + errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
    errh = errh**2 + errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
    errl = np.sqrt(errl); errh = np.sqrt(errh)
  else:
    errl, errh = errors_poisson(counts)

  # Normalize if asked so
  if density:
    counts = counts/norm; errl = errl/norm;  errh = errh/norm

  # Construct the ipo-object
  result = ipo(**{**{'counts':counts,
                     'edges':edges, 'bins':bincs, 'cmbins': cmbins,
                     'weights': weights, 'norm': norm,
                     'density': density, 'nob': bins,
                     'errl': errl, 'errh': errh,
                    },
                  **kwargs})
  return result



def compare_hist(data, weights=[None, None], density=False, **kwargs):
  """
  This function compares to histograms in data=[ref, obj] with(/out) weights
  It returns two hisrogram ipo-objects, obj one with pulls, and both of them
  normalized to one.
  """
  ref = hist(data[0], density=False, **kwargs, weights=weights[0])
  obj = hist(data[1], density=False, **kwargs, weights=weights[1])
  ref_norm = 1; obj_norm = 1;
  if density:
    ref_norm = 1/ref.counts.sum(); obj_norm = 1/obj.counts.sum();
  ref.counts = ref.counts*ref_norm; ref.errl *= ref_norm; ref.errh *= ref_norm
  obj.counts = obj.counts*obj_norm; obj.errl *= obj_norm; obj.errh *= obj_norm
  obj.add('pulls', pull_hist(ref.counts, obj.counts, obj.errl, obj.errh))
  return ref, obj
