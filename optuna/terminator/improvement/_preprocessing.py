import abc
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

import optuna
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import _is_distribution_log
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import intersection_search_space
from optuna.trial._state import TrialState


class BasePreprocessing(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        pass


class PreprocessingPipeline(BasePreprocessing):
    def __init__(self, processes: List[BasePreprocessing]) -> None:
        self._processes = processes

    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        for p in self._processes:
            trials = p.apply(trials, study_direction)
        return trials


class NullPreprocessing(BasePreprocessing):
    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        return trials


class UnscaleLog(BasePreprocessing):
    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        mapped_trials = []
        for trial in trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE

            params, distributions = {}, {}
            for param_name in trial.params.keys():
                (
                    param_value,
                    param_distribution,
                ) = self._convert_param_value_distribution(
                    trial.params[param_name], trial.distributions[param_name]
                )
                params[param_name] = param_value
                distributions[param_name] = param_distribution

            trial = optuna.create_trial(
                value=trial.value,
                params=params,
                distributions=distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
            )
            mapped_trials.append(trial)

        return mapped_trials

    @staticmethod
    def _convert_param_value_distribution(
        value: Any, distribution: BaseDistribution
    ) -> Tuple[Any, BaseDistribution]:
        if isinstance(distribution, (IntDistribution, FloatDistribution)):
            if _is_distribution_log(distribution):
                value = np.log(value)
                low = np.log(distribution.low)
                high = np.log(distribution.high)

                distribution = FloatDistribution(low=low, high=high)

                return value, distribution

        return value, distribution


class SelectTopTrials(BasePreprocessing):
    def __init__(
        self,
        top_trials_ratio: float,
        min_n_trials: int,
    ) -> None:
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials

    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        trials = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]

        trials = sorted(trials, key=lambda t: cast(float, t.value))
        if study_direction == optuna.study.StudyDirection.MAXIMIZE:
            trials = list(reversed(trials))

        top_n = int(len(trials) * self._top_trials_ratio)
        top_n = max(top_n, self._min_n_trials)
        top_n = min(top_n, len(trials))

        return trials[:top_n]


class ToMinimize(BasePreprocessing):
    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        mapped_trials = []
        for trial in trials:
            if study_direction == optuna.study.StudyDirection.MAXIMIZE:
                value = None if trial.value is None else -trial.value
            else:
                value = trial.value

            trial = optuna.create_trial(
                value=value,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
                state=trial.state,
            )

            mapped_trials.append(trial)

        return mapped_trials


class OneToHot(BasePreprocessing):
    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        mapped_trials = []
        for trial in trials:
            params = {}
            distributions: Dict[str, BaseDistribution] = {}
            for param, distribution in trial.distributions.items():
                if isinstance(distribution, CategoricalDistribution):
                    ir = distribution.to_internal_repr(trial.params[param])
                    values = [1.0 if i == ir else 0.0 for i in range(len(distribution.choices))]
                    for i, v in enumerate(values):
                        key = f"i{i}_{param}"
                        params[key] = v
                        distributions[key] = FloatDistribution(0.0, 1.0)
                else:
                    key = f"i0_{param}"
                    params[key] = trial.params[param]
                    distributions[key] = distribution

            trial = optuna.create_trial(
                value=trial.value,
                params=params,
                distributions=distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
                state=trial.state,
            )

            mapped_trials.append(trial)

        return mapped_trials


class AddRandomInputs(BasePreprocessing):
    def __init__(
        self,
        n_additional_trials: int,
        dummy_value: float = np.nan,
    ) -> None:
        self._n_additional_trials = n_additional_trials
        self._dummy_value = dummy_value
        self._rng = np.random.RandomState()

    def apply(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: Optional[optuna.study.StudyDirection],
    ) -> List[optuna.trial.FrozenTrial]:
        search_space = intersection_search_space(trials)

        additional_trials = []
        for _ in range(self._n_additional_trials):
            params = {}
            for param_name, distribution in search_space.items():
                trans = _SearchSpaceTransform({param_name: distribution})
                trans_params = self._rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
                param_value = trans.untransform(trans_params)[param_name]
                params[param_name] = param_value

            trial = optuna.create_trial(
                value=self._dummy_value,
                params=params,
                distributions=search_space,
                state=TrialState.COMPLETE,
            )

            additional_trials.append(trial)

        return trials + additional_trials
