import itertools

import numpy as np
import pytest

from optuna.samplers.tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers.tpe.sampler import default_weights
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import List  # NOQA


class TestParzenEstimator(object):
    @staticmethod
    @pytest.mark.parametrize(
        "mus, prior, magic_clip, endpoints",
        itertools.product(
            ([], [0.4], [-0.4, 0.4]),  # mus
            (True, False),  # prior
            (True, False),  # magic_clip
            (True, False),  # endpoints
        ),
    )
    def test_calculate_shape_check(mus, prior, magic_clip, endpoints):
        # type: (List[float], bool, bool, bool) -> None

        s_weights, s_mus, s_sigmas = _ParzenEstimator._calculate(
            mus,
            -1.0,
            1.0,
            prior_weight=1.0,
            consider_prior=prior,
            consider_magic_clip=magic_clip,
            consider_endpoints=endpoints,
            weights_func=default_weights,
        )

        # Result contains an additional value for a prior distribution if prior is True or
        # len(mus) == 0 (in this case, prior is always used).
        assert len(s_weights) == len(mus) + int(prior) if len(mus) > 0 else len(mus) + 1
        assert len(s_mus) == len(mus) + int(prior) if len(mus) > 0 else len(mus) + 1
        assert len(s_sigmas) == len(mus) + int(prior) if len(mus) > 0 else len(mus) + 1

    # TODO(ytsmiling): Improve test coverage for weights_func.
    @staticmethod
    @pytest.mark.parametrize(
        "mus, flags, expected",
        [
            [
                [],
                {"prior": False, "magic_clip": False, "endpoints": True},
                {"weights": [1.0], "mus": [0.0], "sigmas": [2.0]},
            ],
            [
                [],
                {"prior": True, "magic_clip": False, "endpoints": True},
                {"weights": [1.0], "mus": [0.0], "sigmas": [2.0]},
            ],
            [
                [0.4],
                {"prior": True, "magic_clip": False, "endpoints": True},
                {"weights": [0.5, 0.5], "mus": [0.0, 0.4], "sigmas": [2.0, 0.6]},
            ],
            [
                [-0.4],
                {"prior": True, "magic_clip": False, "endpoints": True},
                {"weights": [0.5, 0.5], "mus": [-0.4, 0.0], "sigmas": [0.6, 2.0]},
            ],
            [
                [-0.4, 0.4],
                {"prior": True, "magic_clip": False, "endpoints": True},
                {"weights": [1.0 / 3] * 3, "mus": [-0.4, 0.0, 0.4], "sigmas": [0.6, 2.0, 0.6]},
            ],
            [
                [-0.4, 0.4],
                {"prior": True, "magic_clip": False, "endpoints": False},
                {"weights": [1.0 / 3] * 3, "mus": [-0.4, 0.0, 0.4], "sigmas": [0.4, 2.0, 0.4]},
            ],
            [
                [-0.4, 0.4],
                {"prior": False, "magic_clip": False, "endpoints": True},
                {"weights": [0.5, 0.5], "mus": [-0.4, 0.4], "sigmas": [0.8, 0.8]},
            ],
            [
                [-0.4, 0.4, 0.41, 0.42],
                {"prior": False, "magic_clip": False, "endpoints": True},
                {
                    "weights": [0.25, 0.25, 0.25, 0.25],
                    "mus": [-0.4, 0.4, 0.41, 0.42],
                    "sigmas": [0.8, 0.8, 0.01, 0.58],
                },
            ],
            [
                [-0.4, 0.4, 0.41, 0.42],
                {"prior": False, "magic_clip": True, "endpoints": True},
                {
                    "weights": [0.25, 0.25, 0.25, 0.25],
                    "mus": [-0.4, 0.4, 0.41, 0.42],
                    "sigmas": [0.8, 0.8, 0.4, 0.58],
                },
            ],
        ],
    )
    def test_calculate(mus, flags, expected):
        # type: (List[float], Dict[str, bool], Dict[str, List[float]]) -> None

        s_weights, s_mus, s_sigmas = _ParzenEstimator._calculate(
            mus,
            -1.0,
            1.0,
            prior_weight=1.0,
            consider_prior=flags["prior"],
            consider_magic_clip=flags["magic_clip"],
            consider_endpoints=flags["endpoints"],
            weights_func=default_weights,
        )

        # Result contains an additional value for a prior distribution if consider_prior is True.
        np.testing.assert_almost_equal(s_weights, expected["weights"])
        np.testing.assert_almost_equal(s_mus, expected["mus"])
        np.testing.assert_almost_equal(s_sigmas, expected["sigmas"])
