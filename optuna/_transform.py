import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.trial import FrozenTrial


class _Transform:
    def __init__(
        self,
        search_space: Dict[str, BaseDistribution],
        transform_log: bool = True,
    ) -> None:
        assert len(search_space) > 0, "Cannot transform if no distributions are given."

        self._search_space = search_space
        self._transform_log = transform_log
        self._bounds, self._column_to_encoded_columns = self._transform_bounds(search_space)

    @property
    def bounds(self) -> numpy.ndarray:
        return self._bounds

    def transform(self, trials: List[FrozenTrial]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        assert len(trials) > 0, "Cannot transform if no trials are given."

        n_trials = len(trials)
        n_bounds = sum(
            1 if not isinstance(d, CategoricalDistribution) else len(d.choices)
            for d in self._search_space.values()
        )

        trans_params = numpy.zeros((n_trials, n_bounds), dtype=numpy.float64)
        trans_values = numpy.empty((n_trials,), dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            bound_idx = 0
            for name, distribution in self._search_space.items():
                param = trial.params[name]
                if isinstance(distribution, CategoricalDistribution):
                    choice_idx = distribution.to_internal_repr(param)
                    trans_params[trial_idx, bound_idx + choice_idx] = 1
                    bound_idx += len(distribution.choices)
                else:
                    trans_params[trial_idx, bound_idx] = self._transform_param(param, distribution)
                    bound_idx += 1
            trans_values[trial_idx] = trial.value

        return trans_params, trans_values

    def untransform_single_params(self, trans_single_params: numpy.ndarray) -> Dict[str, Any]:
        assert trans_single_params.shape == (self._bounds.shape[0],)

        params = {}

        for i, (name, distribution) in enumerate(self._search_space.items()):
            trans_single_param = trans_single_params[self._column_to_encoded_columns[i]]
            params[name] = self._untransform_single_param(trans_single_param, distribution)

        return params

    def _transform_bounds(
        self, search_space: Dict[str, BaseDistribution]
    ) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        n_bounds = sum(
            1 if not isinstance(d, CategoricalDistribution) else len(d.choices)
            for d in search_space.values()
        )

        bounds = numpy.empty((n_bounds, 2), dtype=numpy.float64)
        column_to_encoded_columns: List[numpy.ndarray] = []

        bound_idx = 0
        for d in search_space.values():
            if isinstance(d, CategoricalDistribution):
                n_choices = len(d.choices)
                bounds[bound_idx : bound_idx + n_choices] = (0, 1)
                column_to_encoded_columns.append(numpy.arange(bound_idx, bound_idx + n_choices))
                bound_idx += n_choices
            elif isinstance(
                d,
                (
                    UniformDistribution,
                    LogUniformDistribution,
                    DiscreteUniformDistribution,
                    IntUniformDistribution,
                    IntLogUniformDistribution,
                ),
            ):
                if isinstance(d, UniformDistribution):
                    b = (
                        self._transform_param(d.low, d),
                        self._transform_param(d.high, d),
                    )
                elif isinstance(d, LogUniformDistribution):
                    b = (
                        self._transform_param(d.low, d),
                        self._transform_param(d.high, d),
                    )
                elif isinstance(d, DiscreteUniformDistribution):
                    half_step = 0.5 * d.q
                    b = (
                        self._transform_param(d.low, d) - half_step,
                        self._transform_param(d.high, d) + half_step,
                    )
                elif isinstance(d, IntUniformDistribution):
                    half_step = 0.5 * d.step
                    b = (
                        self._transform_param(d.low, d) - half_step,
                        self._transform_param(d.high, d) + half_step,
                    )
                elif isinstance(d, IntLogUniformDistribution):
                    b = (
                        self._transform_param(d.low - 0.5, d),
                        self._transform_param(d.high + 0.5, d),
                    )
                else:
                    assert False

                bounds[bound_idx] = b
                column_to_encoded_columns.append(numpy.atleast_1d(bound_idx))
                bound_idx += 1
            else:
                assert False

        assert bound_idx == n_bounds

        return bounds, column_to_encoded_columns

    def _transform_param(self, param: Any, distribution: BaseDistribution) -> float:
        d = distribution
        if isinstance(d, CategoricalDistribution):
            assert False
        elif isinstance(d, UniformDistribution):
            trans_param = float(param)
        elif isinstance(d, LogUniformDistribution):
            trans_param = math.log(param) if self._transform_log else float(param)
        elif isinstance(d, DiscreteUniformDistribution):
            trans_param = float(param)
        elif isinstance(d, IntUniformDistribution):
            trans_param = float(param)
        elif isinstance(d, IntLogUniformDistribution):
            trans_param = math.log(param) if self._transform_log else float(param)
        else:
            assert False
        return trans_param

    def _untransform_single_param(
        self, trans_param: numpy.ndarray, distribution: BaseDistribution
    ) -> Any:
        d = distribution
        if isinstance(d, CategoricalDistribution):
            # Select the highest rated one-hot enconding.
            param = d.to_external_repr(trans_param.argmax())
        elif isinstance(d, UniformDistribution):
            param = float(trans_param.item())
        elif isinstance(d, LogUniformDistribution):
            param = math.exp(trans_param) if self._transform_log else float(trans_param)
        elif isinstance(d, DiscreteUniformDistribution):
            # v may slightly exceed range due to round-off errors.
            param = float(
                min(max(numpy.round((trans_param - d.low) / d.q) * d.q + d.low, d.low), d.high)
            )
        elif isinstance(d, IntUniformDistribution):
            param = int(numpy.round((trans_param - d.low) / d.step) * d.step + d.low)
        elif isinstance(d, IntLogUniformDistribution):
            if self._transform_log:
                param = math.exp(trans_param)
                v = numpy.round(param)
                param = int(min(max(v, d.low), d.high))
            else:
                param = int(trans_param)
        else:
            assert False
        return param
