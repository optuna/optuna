import numpy
from typing import NamedTuple  # NOQA
from typing import Optional  # NOQA
from typing import Callable  # NOQA
from typing import List  # NOQA
from typing import Union  # NOQA
from typing import Tuple  # NOQA

GeneralizedList = Union[List, numpy.ndarray]


class ParzenEstimatorParameters(NamedTuple):
    consider_prior: bool
    prior_weight: Optional[float]
    consider_magic_clip: bool
    consider_endpoints: bool
    weights: Callable[[int], GeneralizedList]


class ParzenEstimator(object):
    def __init__(self,
                 mus,  # type: List[float]
                 low,  # type: float
                 high,  # type: float
                 parameters  # type: ParzenEstimatorParameters
                 ):
        # type: (...) -> None
        s_weights, s_mus, sigmas = ParzenEstimator.__calculate(mus,
                                                               low,
                                                               high,
                                                               parameters.consider_prior,
                                                               parameters.prior_weight,
                                                               parameters.consider_magic_clip,
                                                               parameters.consider_endpoints,
                                                               parameters.weights)
        self.weights = s_weights
        self.mus = s_mus
        self.sigmas = sigmas

    @classmethod
    def __calculate(cls,
                    mus,  # type: List[float]
                    low,  # type: float
                    high,  # type: float
                    consider_prior,  # type: bool
                    prior_weight,  # type: Optional[float]
                    consider_magic_clip,  # type: bool
                    consider_endpoints,  # type: bool
                    weights_func  # type: Callable[[int], GeneralizedList]
                    ):
        # type: (...) -> Tuple[List[float], List[float], List[float]]
        mus = numpy.asarray(mus)
        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)
            if len(mus) == 0:
                sorted_mus = numpy.asarray([prior_mu])
                sigma = numpy.asarray([prior_sigma])
                prior_pos = 0
                order = []
            elif len(mus) == 1:
                if prior_mu < mus[0]:
                    prior_pos = 0
                    sorted_mus = numpy.asarray([prior_mu, mus[0]])
                    sigma = numpy.asarray([prior_sigma, prior_sigma * .5])
                else:
                    prior_pos = 1
                    sorted_mus = numpy.asarray([mus[0], prior_mu])
                    sigma = numpy.asarray([prior_sigma * .5, prior_sigma])
                order = [0]
            else:  # len(mus) >= 2
                # decide where prior is placed
                order = numpy.argsort(mus)
                prior_pos = numpy.searchsorted(mus[order], prior_mu)

                # decide mus
                sorted_mus = numpy.zeros(len(mus) + 1)
                sorted_mus[:prior_pos] = mus[order[:prior_pos]]
                sorted_mus[prior_pos] = prior_mu
                sorted_mus[prior_pos + 1:] = mus[order[prior_pos:]]

                # decide sigmas
                low_sorted_mus_high = numpy.append(sorted_mus, high)
                low_sorted_mus_high = numpy.insert(low_sorted_mus_high, 0, low)
                sigma = numpy.zeros_like(low_sorted_mus_high)
                sigma[1:-1] = numpy.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                            low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
                if not consider_endpoints:
                    sigma[1] = sigma[2] - sigma[1]
                    sigma[-2] = sigma[-2] - sigma[-3]
                sigma = sigma[1:-1]

            # decide weights
            unsorted_weights = weights_func(len(mus))
            sorted_weights = numpy.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = prior_weight
            sorted_weights[prior_pos + 1:] = unsorted_weights[order[prior_pos:]]
            sorted_weights /= sorted_weights.sum()
        else:
            order = numpy.argsort(mus)

            # decide mus
            sorted_mus = mus[order]

            # decide sigmas
            if len(mus) == 0:
                sigma = []
            else:
                low_sorted_mus_high = numpy.append(sorted_mus, high)
                low_sorted_mus_high = numpy.insert(low_sorted_mus_high, 0, low)
                sigma = numpy.zeros_like(low_sorted_mus_high)
                sigma[1:-1] = numpy.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                            low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
                if not consider_endpoints:
                    sigma[1] = sigma[2] - sigma[1]
                    sigma[-2] = sigma[-2] - sigma[-3]
                sigma = sigma[1:-1]

            # decide weights
            unsorted_weights = weights_func(len(mus))
            sorted_weights = unsorted_weights[order]
            sorted_weights /= sorted_weights.sum()

        if consider_magic_clip:
            maxsigma = 1.0 * (high - low)
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(sorted_mus)))
            sigma = numpy.clip(sigma, minsigma, maxsigma)
        else:
            maxsigma = 1.0 * (high - low)
            minsigma = 0.0
            sigma = numpy.clip(sigma, maxsigma, minsigma)

        sorted_weights = list(sorted_weights)
        sorted_mus = list(sorted_mus)
        sigma = list(sigma)
        return sorted_weights, sorted_mus, sigma
