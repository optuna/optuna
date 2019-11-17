import collections
import itertools
import random

from optuna.samplers.base import BaseSampler
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA


class GridSampler(BaseSampler):
    """Sampler using grid search.

    This sampler is based on *relative sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'relative sampling'.

    Example:

        .. code::

            >>> import optuna
            >>>
            >>> def objective(trial):
            >>>     x = trial.suggest_uniform('x', -100, 100)
            >>>     y = trial.suggest_int('y', -100, 100)
            >>>     return x ** 2 + y ** 2
            >>>
            >>> grid = {
            >>>     'x': [-50, 0, 50],
            >>>     'y': [-99, 0, 99]
            >>> }
            >>> study = optuna.create_study(sampler=optuna.samplers.GridSampler(grid))
            >>> study.optimize(objective)

    Note:

        :class:`~optuna.samplers.GridSampler` does not take care of a parameter's quantization
        specified by discrete suggest methods but just samples one of values specified in the
        grid. E.g., in the following code snippet, either of ``-0.5`` or ``0.5`` is sampled as
        ``x`` instead of an integer point.

        .. code::

            >>> import optuna
            >>>
            >>> def objective(trial):
            >>>     # The following suggest method specifies an integer point between -5 and 5.
            >>>     x = trial.suggest_discrete_uniform('x' -5, 5, 1)
            >>>     return x ** 2
            >>>
            >>> # Non-int points are specified in the grid.
            >>> grid = {'x': [-0.5, 0.5]}
            >>> study = optuna.create_study(sampler=optuna.samplers.GridSampler(grid))
            >>> study.optimize(objective)

    Args:
        grid:
            A dictionary whose key and value are a parameter name and the corresponding candidates
            of values, respectively.
    """

    def __init__(self, grid):
        # type: (Dict[str, List[Any]]) -> None

        # todo(g-votte): detect non-identical girds and raise an error in case.

        self._grid = collections.OrderedDict(sorted(grid.items(), key=lambda x: x[0]))
        self._grid_product = list(itertools.product(*self._grid.values()))
        self._param_names = sorted(grid.keys())
        self._n_min_trials = len(self._grid_product)

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
            raise ValueError('All grids have been evaluated. If you want to avoid this error, '
                             'please make sure that unnecessary trials do not run during '
                             'optimization by properly setting `n_trials` in `study.optimize`.')

        # In distributed optimization, multiple workers may simultaneously pick up the same grid.
        # To make the conflict less frequent, the grid is chosen randomly.
        grid_id = random.choice(unvisited_grids)
        study._storage.set_trial_system_attr(trial._trial_id, 'grid_id', grid_id)

        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any

        if param_name not in self._grid:
            message = 'The parameter name, {}, is not found in the given grid.'.format(param_name)
            raise ValueError(message)

        grid_id = trial.system_attrs['grid_id']
        param_value = self._grid_product[grid_id][self._param_names.index(param_name)]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            raise ValueError()  # todo(g-votte): fill in the error message.

        return param_value

    def _get_unvisited_grid_ids(self, study):
        # type: (Study) -> List[int]

        trials = study.trials
        trials = [t for t in trials if t.state.is_finished]
        trials = [t for t in trials if 'grid_id' in t.system_attrs]

        visited_grids = [t.system_attrs['grid_id'] for t in trials]
        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)
