from collections import OrderedDict
import copy
from typing import Dict
from typing import Optional
from typing import Set

import optuna
from optuna.distributions import BaseDistribution
from optuna.study import BaseStudy


class IntersectionSearchSpace(object):
    """A class to calculate the intersection search space of a :class:`~optuna.study.BaseStudy`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).
    """

    def __init__(self):
        self._known_trials = set()  # type: Set[int]
        self._search_space = None  # type: Optional[Dict[str, BaseDistribution]]
        self._study_id = None  # type: Optional[int]

    def calculate(self, study: BaseStudy, ordered_dict: bool = False):
        """Returns the intersection search space of the :class:`~optuna.study.BaseStudy`.

        Args:
            study:
                A study with completed trials.
            ordered_dict:
                A boolean flag determining the return type.
                If :obj:`False`, the returned object will be a :obj:`dict`.
                If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict` sorted by
                keys, i.e. parameter names.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.
        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemortyStorage` is used
            # because `InMemortyStorage.create_new_study` always returns the same study ID (i.e., `0`).
            if self._study_id != study._study_id:
                raise ValueError("`IntersectionSearchSpace` cannot handle multiple studies.")

        for trial in study.get_trials(deepcopy=False):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            if trial.number in self._known_trials:
                continue

            self._known_trials.add(trial.number)

            if self._search_space is None:
                self._search_space = copy.copy(trial.distributions)
                continue

            delete_list = []
            for param_name, param_distribution in self._search_space.items():
                if param_name not in trial.distributions:
                    delete_list.append(param_name)
                elif trial.distributions[param_name] != param_distribution:
                    delete_list.append(param_name)

            for param_name in delete_list:
                del self._search_space[param_name]

        search_space = self._search_space or {}

        if ordered_dict:
            search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))

        return search_space


def intersection_search_space(
    study: BaseStudy, ordered_dict: bool = False
) -> Dict[str, BaseDistribution]:
    """Return the intersection search space of the :class:`~optuna.study.BaseStudy`.

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

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    return IntersectionSearchSpace().calculate(study, ordered_dict=ordered_dict)
