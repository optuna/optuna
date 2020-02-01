import copy
import json

import optuna
from optuna.distributions import dict_to_distribution

if optuna.type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.samplers.base import BaseSampler  # NOQA
    from optuna.samplers.random import RandomSampler  # NOQA
    from optuna.samplers.tpe import TPESampler  # NOQA
    from optuna.study import BaseStudy  # NOQA


def intersection_search_space(study, trial_id=None):
    # type: (BaseStudy, Optional[int]) -> Dict[str, BaseDistribution]
    """Return the intersection search space of the :class:`~optuna.study.BaseStudy`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    search_space = None
    for trial in reversed(study.get_trials(deepcopy=False)):
        if trial.state != optuna.structs.TrialState.COMPLETE:
            continue

        if search_space is None:
            search_space = copy.deepcopy(trial.distributions)
            continue

        delete_list = []
        for param_name, param_distribution in search_space.items():
            if param_name not in trial.distributions:
                delete_list.append(param_name)
            elif trial.distributions[param_name] != param_distribution:
                delete_list.append(param_name)

        for param_name in delete_list:
            del search_space[param_name]

        # Retrieve cache from trial_system_attrs.
        if trial_id is None:
            continue

        json_str = trial.system_attrs.get("intersection_search_space", None)  # type: str
        if json_str is None:
            continue
        json_dict = json.loads(json_str)

        delete_list = []
        cached_search_space = {name: dict_to_distribution(dic) for name, dic in json_dict.items()}
        for param_name in search_space:
            if param_name not in cached_search_space:
                delete_list.append(param_name)
            elif cached_search_space[param_name] != search_space[param_name]:
                delete_list.append(param_name)

        for param_name in delete_list:
            del search_space[param_name]

        json_str = json.dumps({
            name: search_space[name]._asdict() for name in search_space or {}
        })
        study._storage.set_trial_system_attr(
            trial_id, "intersection_search_space", json_str,
        )
        break

    return search_space or {}
