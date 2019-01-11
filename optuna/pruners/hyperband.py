from typing import List  # NOQA

from optuna.pruners.base import BasePruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner
from optuna.storages import BaseStorage  # NOQA


class HyperbandPruner(BasePruner):

    def __init__(self, min_resource=1, reduction_factor=3,
                 min_early_stopping_rate_low=0, min_early_stopping_rate_high=4):
        # type: (int, int, int, int) -> None

        self.pruners = []  # type: List[SuccessiveHalvingPruner]
        self.reduction_factor = reduction_factor
        self.resource_budget = 0
        pruner_count = min_early_stopping_rate_high - min_early_stopping_rate_low + 1

        for i in range(0, pruner_count):
            self.resource_budget += self._bracket_resource_budget(i, pruner_count)

            min_early_stopping_rate = min_early_stopping_rate_low + i
            rung_key_prefix = 'bracket{}_'.format(min_early_stopping_rate)

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
                rung_key_prefix=rung_key_prefix,
            )
            self.pruners.append(pruner)

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        i = self._bracket_index(study_id, trial_id)
        # if step == 0:
        #     print("# {}-{}: bracket-{} (budget={})".format(
        #         study_id, trial_id, i,
        #         self._bracket_resource_budget(i, len(self.pruners))))
        return self.pruners[i].prune(storage, study_id, trial_id, step)

    def _bracket_resource_budget(self, pruner_index, pruner_count):
        # type: (int, int) -> int

        n = self.reduction_factor ** (pruner_count - 1)
        budget = n
        for i in range(pruner_index, pruner_count - 1):
            budget += n/2
        return budget

    def _bracket_index(self, study_id, trial_id):
        # typoe: (int, int) -> int

        n = hash('{}_{}'.format(study_id, trial_id)) % self.resource_budget
        for i in range(len(self.pruners)):
            n -= self._bracket_resource_budget(i, len(self.pruners))
            if n < 0:
                return i

        raise RuntimeError
