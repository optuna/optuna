import numpy
from numpy import ndarray
from typing import Callable
from typing import NamedTuple
from typing import Optional

from optuna import types

if types.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Tuple  # NOQA

EPS = 1e-12


class ParzenEstimatorParameters(
        NamedTuple('_ParzenEstimatorParameters', [
            ('consider_prior', bool),
            ('prior_weight', Optional[float]),
            ('consider_magic_clip', bool),
            ('consider_endpoints', bool),
            ('weights', Callable[[int], ndarray]),
        ])):
    pass


class ParzenEstimator(object):
    def __init__(
            self,
            mus,  # type: ndarray
            low,  # type: float
            high,  # type: float
            parameters  # type: ParzenEstimatorParameters
    ):
        # type: (...) -> None

        s_weights, s_mus, sigmas = ParzenEstimator._calculate(
            mus, low, high, parameters.consider_prior, parameters.prior_weight,
            parameters.consider_magic_clip, parameters.consider_endpoints, parameters.weights)
        self.weights = numpy.asarray(s_weights)
        self.mus = numpy.asarray(s_mus)
        self.sigmas = numpy.asarray(sigmas)

    @classmethod
    def _calculate(
            cls,
            mus,  # type: ndarray
            low,  # type: float
            high,  # type: float
            consider_prior,  # type: bool
            prior_weight,  # type: Optional[float]
            consider_magic_clip,  # type: bool
            consider_endpoints,  # type: bool
            weights_func  # type: Callable[[int], ndarray]
    ):
        # type: (...) -> Tuple[List[float], List[float], List[float]]

        mus = numpy.asarray(mus)
        sigma = numpy.asarray([], dtype=float)
        prior_pos = 0
        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)
            if mus.size == 0:
                sorted_mus = numpy.asarray([prior_mu])
                sigma = numpy.asarray([prior_sigma])
                prior_pos = 0
                order = []  # type: List[int]
            else:  # When mus.size is greater than 0.
                # We decide the place of the  prior.
                order = numpy.argsort(mus).astype(int)
                prior_pos = numpy.searchsorted(mus[order], prior_mu)
                # We decide the mus.
                sorted_mus = numpy.zeros(len(mus) + 1)
                sorted_mus[:prior_pos] = mus[order[:prior_pos]]
                sorted_mus[prior_pos] = prior_mu
                sorted_mus[prior_pos + 1:] = mus[order[prior_pos:]]
        else:
            order = numpy.argsort(mus)
            # We decide the mus.
            sorted_mus = mus[order]

        # We decide the sigma.
        if mus.size > 0:
            low_sorted_mus_high = numpy.append(sorted_mus, high)
            low_sorted_mus_high = numpy.insert(low_sorted_mus_high, 0, low)
            sigma = numpy.zeros_like(low_sorted_mus_high)
            sigma[1:-1] = numpy.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                        low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
            if not consider_endpoints and low_sorted_mus_high.size > 2:
                sigma[1] = low_sorted_mus_high[2] - low_sorted_mus_high[1]
                sigma[-2] = low_sorted_mus_high[-2] - low_sorted_mus_high[-3]
            sigma = sigma[1:-1]

        # We decide the weights.
        unsorted_weights = weights_func(mus.size)
        if consider_prior:
            sorted_weights = numpy.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = prior_weight
            sorted_weights[prior_pos + 1:] = unsorted_weights[order[prior_pos:]]
        else:
            sorted_weights = unsorted_weights[order]
        sorted_weights /= sorted_weights.sum()

        # We adjust the range of the 'sigma' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(sorted_mus)))
        else:
            minsigma = EPS
        sigma = numpy.clip(sigma, minsigma, maxsigma)
        if consider_prior:
            sigma[prior_pos] = prior_sigma

        sorted_weights = list(sorted_weights)
        sorted_mus = list(sorted_mus)
        sigma = list(sigma)
        return sorted_weights, sorted_mus, sigma
