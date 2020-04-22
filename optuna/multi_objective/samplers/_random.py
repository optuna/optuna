from typing import Any
from typing import Dict
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler


@experimental("1.4.0")
class RandomMultiObjectiveSampler(BaseMultiObjectiveSampler):
    """Multi-objective sampler using random sampling.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler`
    for more details of 'independent sampling'.

    Example:

        .. testcode::

            import optuna
            from optuna.multi_objective.samplers import RandomMultiObjectiveSampler

            def objective(trial):
                x = trial.suggest_uniform('x', -5, 5)
                y = trial.suggest_uniform('y', -5, 5)
                return x ** 2, y + 10

            study = optuna.multi_objective.create_study(
                ["minimize", "minimize"],
                sampler=RandomMultiObjectiveSampler()
            )
            study.optimize(objective, n_trials=10)

        Args:
            seed: Seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.infer_relative_search_space(study, trial)  # type: ignore

    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_relative(study, trial, search_space)  # type: ignore

    def sample_independent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_independent(
            study, trial, param_name, param_distribution  # type: ignore
        )
