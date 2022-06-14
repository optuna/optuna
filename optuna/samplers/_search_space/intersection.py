from collections import OrderedDict
import copy
import itertools
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study


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
                If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict`
                sorted by keys, i.e. parameter names.

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

        states_of_interest = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING]

        if self._include_pruned:
            states_of_interest.append(optuna.trial.TrialState.PRUNED)

        trials = study.get_trials(deepcopy=False, states=states_of_interest)

        current_cursor = self._cursor

        def reversed_new_trials_iter(
            finished: bool,
        ) -> Generator[optuna.trial.FrozenTrial, None, None]:
            nonlocal current_cursor
            return (
                trial
                for trial in itertools.takewhile(
                    lambda t: t.number < current_cursor, reversed(trials)
                )
                if trial.state.is_finished() == finished
            )

        dists: Any
        dists = self._search_space.items() if self._search_space is not None else None
        for trial in reversed_new_trials_iter(finished=True):
            if dists is None:
                dists = trial.distributions.items()
            else:
                # We use lambda to capture the value of `trial.distributions`.
                dists = (
                    lambda d: (
                        (param, dist) for param, dist in dists if param in d and dist == d[param]
                    )
                )(trial.distributions)

        self._search_space = dict(dists) if dists is not None else None

        self._cursor = max(
            (trial.number for trial in reversed_new_trials_iter(finished=False)),
            default=-1 if len(trials) == 0 else trials[-1].number,
        )

        search_space = self._search_space or {}

        if ordered_dict:
            search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))

        return copy.deepcopy(search_space)


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
            If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict` sorted by
            keys, i.e. parameter names.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    return IntersectionSearchSpace(include_pruned=include_pruned).calculate(
        study, ordered_dict=ordered_dict
    )
