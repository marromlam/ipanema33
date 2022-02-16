from ..params import Parameters
import numpy as np
import warnings


__all__ = ['compute_sweights', 'sweights_u']


def compute_sweights(model, params, yields, weights=False):
    r"""
    Computes sWeights from probability density functions for different
    components/species in a fit model (for instance signal and background)
    fitted on some data.

    Parameters
    ----------
    model : function
    Fit model.
    params : ipanema.Parameters
      Set of parameters for the model, without the yields.
    yields : ipanema.Parameters
      Set of yield parameters.
    weights : array
      Effective size of the sample to be used.

    Returns
    -------
    dict
    Dictionary with a set of sWeights for each of the species in yields.

    """

    # Evaluate full model
    prob_model = model(**params.valuesdict(), **yields.valuesdict())
    # print(prob_model.allocator)

    # check for weights
    if not weights:
        weights = np.ones_like(prob_model)
    else:
        try:
            weights = weights.get()
        except:
            weights = weights

    # Get number of events and yield size
    _sum_yields = np.sum([v.uvalue.n for v in yields.values()])
    _number_of_events = len(prob_model)

    # Dictionary with species and sWeights
    sweights = {y: np.zeros(_number_of_events) for y in yields}

    # compute statistical power of the sample
    stat_power = np.sum(weights)
    # stat_power = _number_of_events
    # print("Stat power is:", stat_power)

    # Create as many sets of parameters as species. Each one of them only turns
    # on a specie, and set the others to zero
    _yields = {}
    for k, v in yields.items():
        if not np.allclose(v.value/_sum_yields, 0, 1e-10):
            __yields = Parameters.clone(yields)
            for _k in __yields.keys():
                __yields[_k].set(value=0, init=0, min=-np.inf, max=np.inf)
            __yields[k].set(value=1/stat_power, init=1)
            _yields.update({k: __yields})
        else:
            msg = f"Specie {k} is compatible with zero. Flushing its sWeights to zero."
            warnings.warn(msg)

    # Stack all
    if len(_yields) > 1:
        prob_species = np.vstack(
            [model(**params.valuesdict(), **y.valuesdict()) for y in _yields.values()]).T
        prob_species_norm = prob_species / prob_model[:, None]

        # Sanity check
        MLSR = prob_species_norm.sum(axis=0)

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
        # print(prob_species_norm, weights)
        Vinv = (weights[:, None]*prob_species_norm).T.dot(prob_species_norm)
        V = np.linalg.inv(Vinv)

        # Compute the set of sweights
        _sweights = prob_species.dot(V) / prob_model[:, None]

        # Fill sweights Dictionary with the computed sWeights
        for i, y in enumerate(_yields):
            sweights[y] = _sweights[:, i] * weights
    else:
        # since there is only one specie, we just need to fill it with ones
        sweights[list(yields.keys())[0]] = weights

    return sweights


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


# vim:fdm=marker
