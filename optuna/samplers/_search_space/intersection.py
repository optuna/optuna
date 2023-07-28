import copy
from typing import Dict
from typing import Optional

import optuna
from optuna._deprecated import deprecated_class
from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.study import Study


@deprecated_class(
    "3.2.0",
    "4.0.0",
    name="optuna.samplers.IntersectionSearchSpace",
    text="Please use optuna.search_space.IntersectionSearchSpace instead.",
)
class IntersectionSearchSpace:
    """A class to calculate the intersection search space of a :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Note that an instance of this class is supposed to be used for only one study.
    If different studies are passed to :func:`~optuna.samplers.IntersectionSearchSpace.calculate`,
    a :obj:`ValueError` is raised.

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cursor: int = -1
        self._search_space: Optional[Dict[str, BaseDistribution]] = None
        self._study_id: Optional[int] = None

        self._include_pruned = include_pruned

    def calculate(self, study: Study, ordered_dict: bool = False) -> Dict[str, BaseDistribution]:
        """Returns the intersection search space of the :class:`~optuna.study.Study`.

        Args:
            study:
                A study with completed trials. The same study must be passed for one instance
                of this class through its lifetime.
            ordered_dict:
                A boolean flag determining the return type.
                If :obj:`False`, the returned object will be a :obj:`dict`.
                If :obj:`True`, the returned object will be a :obj:`dict` sorted by keys, i.e.
                parameter names.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`IntersectionSearchSpace` cannot handle multiple studies.")

        states_of_interest = [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
        ]

        if self._include_pruned:
            states_of_interest.append(optuna.trial.TrialState.PRUNED)

        trials = study._get_trials(deepcopy=False, states=states_of_interest, use_cache=False)

        next_cursor = trials[-1].number + 1 if len(trials) > 0 else -1
        for trial in reversed(trials):
            if self._cursor > trial.number:
                break

            if not trial.state.is_finished():
                next_cursor = trial.number
                continue

            if self._search_space is None:
                self._search_space = copy.copy(trial.distributions)
                continue

            self._search_space = {
                name: distribution
                for name, distribution in self._search_space.items()
                if trial.distributions.get(name) == distribution
            }

        self._cursor = next_cursor
        search_space = self._search_space or {}

        if ordered_dict:
            search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))

        return copy.deepcopy(search_space)


@deprecated_func(
    "3.2.0",
    "4.0.0",
    name="optuna.samplers.intersection_search_space",
    text="Please use optuna.search_space.intersection_search_space instead.",
)
def intersection_search_space(
    study: Study, ordered_dict: bool = False, include_pruned: bool = False
) -> Dict[str, BaseDistribution]:
    """Return the intersection search space of the :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    .. note::
        :class:`~optuna.samplers.IntersectionSearchSpace` provides the same functionality with
        a much faster way. Please consider using it if you want to reduce execution time
        as much as possible.

    Args:
        study:
            A study with completed trials.
        ordered_dict:
            A boolean flag determining the return type.
            If :obj:`False`, the returned object will be a :obj:`dict`.
            If :obj:`True`, the returned object will be a :obj:`dict` sorted by keys, i.e.
            parameter names.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    return IntersectionSearchSpace(include_pruned=include_pruned).calculate(
        study, ordered_dict=ordered_dict
    )
