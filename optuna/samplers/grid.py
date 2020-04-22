import collections
import itertools
import random

from optuna._experimental import experimental
from optuna.samplers.base import BaseSampler
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Mapping  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA

    GridValueType = Union[str, float, int, bool, None]


@experimental("1.2.0")
class GridSampler(BaseSampler):
    """Sampler using grid search.

    With :class:`~optuna.samplers.GridSampler`, the trials suggest all combinations of parameters
    in the given search space during the study.

    Example:

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_int('y', -100, 100)
                return x ** 2 + y ** 2

            search_space = {
                'x': [-50, 0, 50],
                'y': [-99, 0, 99]
            }
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective, n_trials=3*3)

    Note:

        :class:`~optuna.samplers.GridSampler` raises an error if all combinations in the passed
        ``search_space`` has already been evaluated. Please make sure that unnecessary trials do
        not run during optimization by properly setting ``n_trials`` in the
        :func:`~optuna.study.Study.optimize` method.

    Note:

        :class:`~optuna.samplers.GridSampler` does not take care of a parameter's quantization
        specified by discrete suggest methods but just samples one of values specified in the
        search space. E.g., in the following code snippet, either of ``-0.5`` or ``0.5`` is
        sampled as ``x`` instead of an integer point.

        .. testcode::

            import optuna

            def objective(trial):
                # The following suggest method specifies integer points between -5 and 5.
                x = trial.suggest_discrete_uniform('x', -5, 5, 1)
                return x ** 2

            # Non-int points are specified in the grid.
            search_space = {'x': [-0.5, 0.5]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective, n_trials=2)

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding candidates
            of values, respectively.
    """

    def __init__(self, search_space):
        # type: (Mapping[str, Sequence[GridValueType]]) -> None

        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = collections.OrderedDict()
        for param_name, param_values in sorted(search_space.items(), key=lambda x: x[0]):
            self._search_space[param_name] = sorted(param_values)

        self._all_grids = list(itertools.product(*self._search_space.values()))
        self._param_names = sorted(search_space.keys())
        self._n_min_trials = len(self._all_grids)

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        return {}

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        # Instead of returning param values, GridSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        unvisited_grids = self._get_unvisited_grid_ids(study)

        if len(unvisited_grids) == 0:
            raise ValueError(
                "All grids have been evaluated. If you want to avoid this error, "
                "please make sure that unnecessary trials do not run during "
                "optimization by properly setting `n_trials` in `study.optimize`."
            )

        # In distributed optimization, multiple workers may simultaneously pick up the same grid.
        # To make the conflict less frequent, the grid is chosen randomly.
        grid_id = random.choice(unvisited_grids)

        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any

        if param_name not in self._search_space:
            message = "The parameter name, {}, is not found in the given grid.".format(param_name)
            raise ValueError(message)

        # TODO(c-bata): Reduce the number of duplicated evaluations on multiple workers.
        # Current selection logic may evaluate the same parameters multiple times.
        # See https://gist.github.com/c-bata/f759f64becb24eea2040f4b2e3afce8f for details.
        grid_id = trial.system_attrs["grid_id"]
        param_value = self._all_grids[grid_id][self._param_names.index(param_name)]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            raise ValueError(
                "The value `{}` is out of range of the parameter `{}`. Please make "
                "sure the search space of the `GridSampler` only contains values "
                "consistent with the distribution specified in the objective "
                "function. The distribution is: `{}`.".format(
                    param_value, param_name, param_distribution
                )
            )

        return param_value

    @staticmethod
    def _check_value(param_name, param_value):
        # type: (str, Any) -> None

        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        raise ValueError(
            "{} contains a value with the type of {}, which is not supported by "
            "`GridSampler`. Please make sure a value is `str`, `int`, `float`, `bool`"
            " or `None`.".format(param_name, type(param_value))
        )

    def _get_unvisited_grid_ids(self, study):
        # type: (Study) -> List[int]

        # List up unvisited grids based on already finished ones.
        visited_grids = []
        for t in study.trials:
            if (
                t.state.is_finished()
                and "grid_id" in t.system_attrs
                and self._same_search_space(t.system_attrs["search_space"])
            ):
                visited_grids.append(t.system_attrs["grid_id"])

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)

    def _same_search_space(self, search_space):
        # type: (Mapping[str, Sequence[GridValueType]]) -> bool

        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            for i, param_value in enumerate(sorted(search_space[param_name])):
                if param_value != self._search_space[param_name][i]:
                    return False

        return True
