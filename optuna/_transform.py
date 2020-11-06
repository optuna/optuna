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
        # TODO(hvy): Introduce `one_hot: bool` parameter to control whether categorical should be
        # one-hot encoded. This could be useful for making this class more general and usable from
        # e.g. the `RandomSampler`, where we don't necessarily have to one-hot encode.
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
                    choice_idx = int(self._transform_param(param, distribution))
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
        for distribution in search_space.values():
            if isinstance(distribution, CategoricalDistribution):
                n_choices = len(distribution.choices)
                bounds[bound_idx : bound_idx + n_choices] = [0, 1]
                column_to_encoded_columns.append(numpy.arange(bound_idx, bound_idx + n_choices))
                bound_idx += n_choices
            elif isinstance(
                distribution,
                (
                    UniformDistribution,
                    LogUniformDistribution,
                    DiscreteUniformDistribution,
                    IntUniformDistribution,
                    IntLogUniformDistribution,
                ),
            ):
                bounds[bound_idx] = self._transform_distribution_bounds(distribution)
                column_to_encoded_columns.append(numpy.atleast_1d(bound_idx))
                bound_idx += 1
        assert bound_idx == n_bounds

        return bounds, column_to_encoded_columns

    def _transform_param(self, param: Any, distribution: BaseDistribution) -> float:
        d = distribution
        if isinstance(d, CategoricalDistribution):
            trans_param = float(d.to_internal_repr(param))
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

    def _transform_distribution_bounds(
        self, distribution: BaseDistribution
    ) -> Tuple[float, float]:
        if isinstance(distribution, CategoricalDistribution):
            bounds = (0.0, float(len(distribution.choices)))
        elif isinstance(
            distribution,
            (
                UniformDistribution,
                LogUniformDistribution,
                DiscreteUniformDistribution,
                IntUniformDistribution,
                IntLogUniformDistribution,
            ),
        ):
            # TODO(hvy): Allow subtracting eps from high to use this class from `CmaEsSampler`.
            # TODO(hvy): Subtract 0.5 * steps for distributions with steps.
            bounds = (
                self._transform_param(distribution.low, distribution),
                self._transform_param(distribution.high, distribution),
            )
        else:
            assert False
        return bounds

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
            v = numpy.round(trans_param / d.q) * d.q + d.low
            # v may slightly exceed range due to round-off errors.
            param = float(min(max(v, d.low), d.high))
        elif isinstance(d, IntUniformDistribution):
            r = numpy.round((trans_param - d.low) / d.step)
            v = r * d.step + d.low
            param = int(v)
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
