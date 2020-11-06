from collections import OrderedDict
import math
import sys
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
        transform_step: bool = True,
    ) -> None:
        # In Python 3.6, dictionary orders are not guaranteed. The order must be fixed throughout
        # since the columns in the transformed representation must be correctly mapped back to the
        # parameters and distributions.
        if sys.version_info.major == 3 and sys.version_info.minor < 7:
            search_space = OrderedDict(search_space)

        bounds, column_to_encoded_columns = _transform_bounds(
            search_space, transform_log, transform_step
        )

        self._bounds = bounds
        self._column_to_encoded_columns = column_to_encoded_columns
        self._search_space = search_space
        self._transform_log = transform_log

    @property
    def bounds(self) -> numpy.ndarray:
        return self._bounds

    def transform(self, trials: List[FrozenTrial]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return _transform_params_and_values(trials, self._search_space, self._transform_log)

    def untransform_single_params(self, trans_single_params: numpy.ndarray) -> Dict[str, Any]:
        assert trans_single_params.shape == (self._bounds.shape[0],)

        params = {}

        for i, (name, distribution) in enumerate(self._search_space.items()):
            trans_single_param = trans_single_params[self._column_to_encoded_columns[i]]
            params[name] = _untransform_single_param(
                trans_single_param, distribution, self._transform_log
            )

        return params


def _transform_bounds(
    search_space: Dict[str, BaseDistribution], transform_log: bool, transform_step: bool
) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
    assert len(search_space) > 0, "Cannot transform if no distributions are given."

    n_bounds = sum(
        len(d.choices) if isinstance(d, CategoricalDistribution) else 1
        for d in search_space.values()
    )

    bounds = numpy.empty((n_bounds, 2), dtype=numpy.float64)
    column_to_encoded_columns: List[numpy.ndarray] = []

    bound_idx = 0
    for distribution in search_space.values():
        d = distribution
        if isinstance(d, CategoricalDistribution):
            n_choices = len(d.choices)
            bounds[bound_idx : bound_idx + n_choices] = (0, 1)  # Broadcasted across all choices.
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
                bds = (
                    _transform_param(d.low, d, transform_log),
                    _transform_param(d.high, d, transform_log),
                )
            elif isinstance(d, LogUniformDistribution):
                bds = (
                    _transform_param(d.low, d, transform_log),
                    _transform_param(d.high, d, transform_log),
                )
            elif isinstance(d, DiscreteUniformDistribution):
                half_step = 0.5 * d.q if transform_step else 0.0
                bds = (
                    _transform_param(d.low, d, transform_log) - half_step,
                    _transform_param(d.high, d, transform_log) + half_step,
                )
            elif isinstance(d, IntUniformDistribution):
                half_step = 0.5 * d.step if transform_step else 0.0
                bds = (
                    _transform_param(d.low, d, transform_log) - half_step,
                    _transform_param(d.high, d, transform_log) + half_step,
                )
            elif isinstance(d, IntLogUniformDistribution):
                half_step = 0.5 if transform_step else 0.0
                bds = (
                    _transform_param(d.low - half_step, d, transform_log),
                    _transform_param(d.high + half_step, d, transform_log),
                )
            else:
                assert False

            bounds[bound_idx] = bds
            column_to_encoded_columns.append(numpy.atleast_1d(bound_idx))
            bound_idx += 1
        else:
            assert False

    assert bound_idx == n_bounds

    return bounds, column_to_encoded_columns


def _transform_params_and_values(
    trials: List[FrozenTrial], search_space: Dict[str, BaseDistribution], transform_log: bool
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    assert len(trials) > 0, "Cannot transform if no trials are given."

    n_trials = len(trials)
    n_bounds = sum(
        len(d.choices) if isinstance(d, CategoricalDistribution) else 1
        for d in search_space.values()
    )

    trans_params = numpy.zeros((n_trials, n_bounds), dtype=numpy.float64)
    trans_values = numpy.empty((n_trials,), dtype=numpy.float64)

    for trial_idx, trial in enumerate(trials):
        bound_idx = 0
        for name, distribution in search_space.items():
            param = trial.params[name]
            if isinstance(distribution, CategoricalDistribution):
                choice_idx = distribution.to_internal_repr(param)
                trans_params[trial_idx, bound_idx + choice_idx] = 1
                bound_idx += len(distribution.choices)
            else:
                trans_params[trial_idx, bound_idx] = _transform_param(
                    param, distribution, transform_log
                )
                bound_idx += 1
        trans_values[trial_idx] = trial.value

    return trans_params, trans_values


def _transform_param(param: Any, distribution: BaseDistribution, transform_log: bool) -> float:
    d = distribution

    if isinstance(d, CategoricalDistribution):
        assert False
    elif isinstance(d, UniformDistribution):
        trans_param = float(param)
    elif isinstance(d, LogUniformDistribution):
        trans_param = math.log(param) if transform_log else float(param)
    elif isinstance(d, DiscreteUniformDistribution):
        trans_param = float(param)
    elif isinstance(d, IntUniformDistribution):
        trans_param = float(param)
    elif isinstance(d, IntLogUniformDistribution):
        trans_param = math.log(param) if transform_log else float(param)
    else:
        assert False

    return trans_param


def _untransform_single_param(
    trans_param: numpy.ndarray, distribution: BaseDistribution, transform_log: bool
) -> Any:
    d = distribution

    if isinstance(d, CategoricalDistribution):
        # Select the highest rated one-hot encoding.
        param = d.to_external_repr(trans_param.argmax())
    elif isinstance(d, UniformDistribution):
        param = float(trans_param.item())
    elif isinstance(d, LogUniformDistribution):
        param = math.exp(trans_param) if transform_log else float(trans_param)
    elif isinstance(d, DiscreteUniformDistribution):
        # v may slightly exceed range due to round-off errors.
        param = float(
            min(max(numpy.round((trans_param - d.low) / d.q) * d.q + d.low, d.low), d.high)
        )
    elif isinstance(d, IntUniformDistribution):
        param = int(numpy.round((trans_param - d.low) / d.step) * d.step + d.low)
    elif isinstance(d, IntLogUniformDistribution):
        if transform_log:
            param = math.exp(trans_param)
            v = numpy.round(param)
            param = int(min(max(v, d.low), d.high))
        else:
            param = int(trans_param)
    else:
        assert False

    return param
