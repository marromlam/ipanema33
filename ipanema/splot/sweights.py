from ..params import Parameters
import numpy as np
import warnings


__all__ = ['compute_sweights', 'sweights_u']


def compute_sweights(model, params, yields):
  r"""
  Computes sWeights from probability density functions for different components/species in
  a fit model (for instance signal and background) fitted on some data.

  Parameters
  ----------
  model : function
  Fit model.
  data : ristra
  Dataset.
  params : ipanema.Parameters
  Set of parameters for the model, without the yields.
  yields : ipanema.Parameters
  Set of yield parameters. 

  Returns
  -------
  dict
  Dictionary with a set of sWeights for each of the species in yields.
  
  """

  # Evaluate full model
  full_model = model(**params.valuesdict(), **yields.valuesdict())

  # Create as many sets of parameters as species. Each one of them only turns on
  # a specie, and set the others to zero
  _yields = []
  _sum_yields = np.sum([v.uvalue.n for v in yields.values()])
  for k,v in yields.items():
      __yields = Parameters.clone(yields)
      for _k in __yields.keys():
        __yields[_k].set(value=0, init=0, min=-np.inf, max=np.inf)
      __yields[k].set(value=1/len(full_model), init=1)
      _yields.append(__yields)

  # Stack all  
  p = np.vstack([model(**params.valuesdict(), **y.valuesdict()) for y in _yields]).T
  pN = p / full_model[:, None]

  # Sanity check
  MLSR = pN.sum(axis=0)

  def warning_message(tolerance):
    msg = "The Maximum Likelihood Sum Rule sanity check, described in equation 17 of"
    msg += " arXiv:physics/0402083, failed.\n"
    msg += " According to this check the following quantities\n"
    for y, mlsr in zip(yields, MLSR):
      msg += f"\t* {y}: {mlsr},\n"
    msg += f"should be equal to 1.0 with an absolute tolerance of {tolerance}."
    return msg

  if not np.allclose(MLSR, 1, atol=5e-2):
    msg = " The numbers suggest that the model is not fitted to the data."
    msg += " Please check your fit."
    warnings.warn(msg)
  elif not np.allclose(MLSR, 1, atol=5e-3):
    msg = " If the fit to the data is good please ignore this warning."
    warnings.warn(msg)

  # Get correlation matrix
  Vinv = (pN).T.dot(pN)
  V = np.linalg.inv(Vinv)
  
  # Compute the set of sweights
  sweights = p.dot(V) / full_model[:, None]

  return {y: sweights[:, i] for i, y in enumerate(yields)}


def sweights_u(arr, weights, *args, **kwargs):
  r'''
  Get the uncertainty associated to the sWeights related to some array.
  Arguments are same as :func:`numpy.histogram`.
  By definition, the uncertainty on the s-weights (for plotting), is defined
  as the sum of the squares of the weights in that bin, like
  .. math:: \sigma = \sqrt{\sum_{b \in \delta x} \omega^2}

  Parameters
  ----------
  arr : numpy.ndarray
  Array of data.
  weights : numpy.ndarray
  Set of sWeights.

  Return
  ------
  numpy.ndarray
  Set of uncertainties associated to sWeights.
  '''
  return np.sqrt(np.histogram(arr, weights=weights*weights, *args, **kwargs)[0])


#vim:foldmethod=marker
