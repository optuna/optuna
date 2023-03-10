from collections import OrderedDict
import copy
from typing import Dict
from typing import List

import optuna
from optuna.distributions import BaseDistribution


# TODO(g-votte): delete this class as it duplicates with `optuna.samplers._search_space`
# TODO(g-votte): to do so, we may want to extract the original module as an upper one
class IntersectionSearchSpace:
    def calculate(self, trials: List[optuna.trial.FrozenTrial]) -> Dict[str, BaseDistribution]:
        states_of_interest = [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
        ]
        trials = [trial for trial in trials if trial.state in states_of_interest]

        if len(trials) == 0:
            return OrderedDict({})

        search_space = copy.copy(trials[-1].distributions)

        for trial in reversed(trials):
            search_space = {
                name: distribution
                for name, distribution in search_space.items()
                if trial.distributions.get(name) == distribution
            }

        search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))

        return copy.deepcopy(search_space)
