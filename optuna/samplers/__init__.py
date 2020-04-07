from collections import OrderedDict
import copy
import json

import optuna
from optuna.distributions import dict_to_distribution
from optuna.distributions import distribution_to_dict
from optuna.samplers.base import BaseSampler  # NOQA
from optuna.samplers.cmaes import CmaEsSampler  # NOQA
from optuna.samplers.grid import GridSampler  # NOQA
from optuna.samplers.random import RandomSampler  # NOQA
from optuna.samplers.tpe import TPESampler  # NOQA

if optuna.type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import BaseStudy  # NOQA


def intersection_search_space(study, ordered_dict=False, trial_id=None):
    # type: (BaseStudy, bool, Optional[int]) -> Dict[str, BaseDistribution]
    """Return the intersection search space of the :class:`~optuna.study.BaseStudy`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Args:
        study:
            A study with completed trials.
        ordered_dict:
            A boolean flag determining the return type.
            If :obj:`False`, the returned object will be a :obj:`dict`.
            If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict` sorted by
            keys, i.e. parameter names.
        trial_id:
            A trial id for building and retrieving an intersection_search_space cache.
            If you set trial_id, this function will be faster.

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    search_space = None  # type: Optional[Dict[str, BaseDistribution]]

    # **How an `intersection_search_space` cache accelerate this function?**
    #
    # | -------- | -------- | ------------------- | ------------------------- |
    # | trial_id | status   | search_space        | intersection_search_space |
    # | -------- | -------- | ------------------- | ------------------------- |
    # |        1 | COMPLETE | {x1: ..., x2: ... } | {x1: ..., x2: ... }       |
    # |        2 | COMPLETE | {x1: ... }          | {x1: ... }                |
    # ~          ~          ~                     ~                           ~
    # |       50 | COMPLETE | {x1: ... }          | {x1: ... }                |
    # |       51 | COMPLETE | {x1: ..., x2: ... } |                           |
    # |       52 | COMPLETE | {x1: ... }          |                           |
    # |       53 | RUNNING  |                     |                           |
    # | -------- | -------- | ------------------- | ------------------------- |
    #
    # Now we assume that the above trials are store in the storage.
    # `intersection_search_space(study, trial_id=53)` should return `{x1: ...}`.
    #
    # We iterates completed trials from the end of trials.
    # In this case, we calculates an intersection of the following search spaces.
    #
    # 1. `{x1: ...}` - search_space of trial_id=52
    # 2. `{x1: ..., x2: ...}` - search_space of trial_id=51
    # 3. `{x1: ...}` - intersection_search_space cache of trial_id=50
    #
    # Before returning the function, we build an intersection_search_space cache in trial_id=53.

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

        # Retrieve an intersection_search_space cache.
        if trial_id is None:
            continue

        json_str = trial.system_attrs.get("intersection_search_space", None)  # type: Optional[str]
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

        break

    if trial_id is not None and search_space is not None:
        # Store an intersection_search_space cache.
        json_str = json.dumps(
            {name: distribution_to_dict(search_space[name]) for name in search_space}
        )
        study._storage.set_trial_system_attr(
            trial_id, "intersection_search_space", json_str,
        )

    search_space = search_space or {}
    if ordered_dict:
        search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))
    return search_space
