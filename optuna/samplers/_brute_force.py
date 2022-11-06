import copy
import decimal
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class BruteForceSampler(BaseSampler):
    """Sampler using brute force.

    This sampler performs exhaustive search on the defined search space.

    Note:
        The defined search space must be finite. Therefore, when using
        :class:`~optuna.distributions.FloatDistibution`, `step=None` is not allowed.

    Note:
        This sampler should not be used in combination with other samplers. For example,
        in distributed optimization, if :class:`~optuna.samplers.BruteForceSampler` and another
        sampler run optimizations simultaneously, complete exhaustive search may not be performed.

    Note:
        The objective function must be fixed during the search. Otherwise, sampler may fail to
        try the entire search space.

    Args:
        seed:
            A seed to fix the order of trials as the search order randomly shuffled. Please note
            that it is not recommended using this option in distributed optimization settings since
            this option cannot ensure the order of trials and may increase the number of duplicate
            suggestions during distributed optimization.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        candidates = _enumerate_candidates(param_distribution)
        if len(candidates) == 0:
            raise ValueError("empty candidates")

        self._rng.shuffle(candidates)

        for value in candidates[1:]:
            params = copy.deepcopy(trial.params)
            params[param_name] = value
            study.enqueue_trial(params, skip_if_exists=True)

        return candidates[0]

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if len(study.get_trials(deepcopy=False, states=(TrialState.WAITING,))) == 0:
            study.stop()


def _enumerate_candidates(param_distribution: BaseDistribution) -> Sequence[Any]:
    if isinstance(param_distribution, FloatDistribution):
        if param_distribution.step is None:
            raise ValueError(
                "FloatDistribution.step must be given for BruteForceSampler"
                " (otherwise, the search space will be infinite)"
            )
        low = decimal.Decimal(str(param_distribution.low))
        high = decimal.Decimal(str(param_distribution.high))
        step = decimal.Decimal(str(param_distribution.step))

        ret = []
        value = low
        while value <= high:
            ret.append(float(value))
            value += step

        return ret
    elif isinstance(param_distribution, IntDistribution):
        return list(
            range(param_distribution.low, param_distribution.high + 1, param_distribution.step)
        )
    elif isinstance(param_distribution, CategoricalDistribution):
        return list(param_distribution.choices)
    else:
        raise ValueError("unknown distribution")
