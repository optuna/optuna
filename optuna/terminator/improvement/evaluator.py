import abc
from typing import Dict
from typing import List
from typing import Optional

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.search_space import intersection_search_space
from optuna.study import StudyDirection
from optuna.terminator.improvement._preprocessing import AddRandomInputs
from optuna.terminator.improvement._preprocessing import BasePreprocessing
from optuna.terminator.improvement._preprocessing import OneToHot
from optuna.terminator.improvement._preprocessing import PreprocessingPipeline
from optuna.terminator.improvement._preprocessing import SelectTopTrials
from optuna.terminator.improvement._preprocessing import ToMinimize
from optuna.terminator.improvement._preprocessing import UnscaleLog
from optuna.terminator.improvement.gp.base import _min_lcb
from optuna.terminator.improvement.gp.base import _min_ucb
from optuna.terminator.improvement.gp.base import BaseGaussianProcess
from optuna.terminator.improvement.gp.botorch import _BoTorchGaussianProcess
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


@experimental_class("3.2.0")
class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    """Base class for improvement evaluators."""

    @abc.abstractmethod
    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        pass


@experimental_class("3.2.0")
class RegretBoundEvaluator(BaseImprovementEvaluator):
    """An error evaluator for upper bound on the regret with high-probability confidence.

    This evaluator evaluates the regret of current best solution, which defined as the difference
    between the objective value of the best solution and of the global optimum. To be specific,
    this evaluator calculates the upper bound on the regret based on the fact that empirical
    estimator of the objective function is bounded by lower and upper confidence bounds with
    high probability under the Gaussian process model assumption.

    Args:
        gp:
            A Gaussian process model on which evaluation base. If not specified, the default
            Gaussian process model is used.
        top_trials_ratio:
            A ratio of top trials to be considered when estimating the regret. Default to 0.5.
        min_n_trials:
            A minimum number of complete trials to estimate the regret. Default to 20.
        min_lcb_n_additional_samples:
            A minimum number of additional samples to estimate the lower confidence bound.
            Default to 2000.
    """

    def __init__(
        self,
        gp: Optional[BaseGaussianProcess] = None,
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        min_lcb_n_additional_samples: int = 2000,
    ) -> None:
        self._gp = gp or _BoTorchGaussianProcess()
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials
        self._min_lcb_n_additional_samples = min_lcb_n_additional_samples

    def get_preprocessing(self, add_random_inputs: bool = False) -> BasePreprocessing:
        processes = [
            SelectTopTrials(
                top_trials_ratio=self._top_trials_ratio,
                min_n_trials=self._min_n_trials,
            ),
            UnscaleLog(),
            ToMinimize(),
        ]

        if add_random_inputs:
            processes += [AddRandomInputs(self._min_lcb_n_additional_samples)]

        processes += [OneToHot()]

        return PreprocessingPipeline(processes)

    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        search_space = intersection_search_space(trials)
        self._validate_input(trials, search_space)

        fit_trials = self.get_preprocessing().apply(trials, study_direction)
        lcb_trials = self.get_preprocessing(add_random_inputs=True).apply(trials, study_direction)

        n_params = len(search_space)
        n_trials = len(fit_trials)

        self._gp.fit(fit_trials)

        ucb = _min_ucb(trials=fit_trials, gp=self._gp, n_params=n_params, n_trials=n_trials)
        lcb = _min_lcb(trials=lcb_trials, gp=self._gp, n_params=n_params, n_trials=n_trials)

        regret_bound = ucb - lcb

        return regret_bound

    @classmethod
    def _validate_input(
        cls, trials: List[FrozenTrial], search_space: Dict[str, BaseDistribution]
    ) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the regret bound cannot be evaluated."
            )

        if len(search_space) == 0:
            raise ValueError(
                "The intersection search space is empty. This condition is not supported by "
                f"{cls.__name__}."
            )


@experimental_class("3.4.0")
class BestValueStagnationEvaluator(BaseImprovementEvaluator):
    """Evaluates the stagnation period of the best value in an optimization process.

    This class is initialized with a maximum stagnation period (`max_stagnation_trials`)
    and is designed to evaluate the remaining trials before reaching this maximum period
    of allowed stagnation. If this remaining trials reach zero, the trial terminates.
    Therefore, the default error evaluator is instantiated by StaticErrorEvaluator(const=0).

    Args:
        max_stagnation_trials:
            The maximum number of trials allowed for stagnation.
    """

    def __init__(
        self,
        max_stagnation_trials: int = 30,
    ) -> None:
        if max_stagnation_trials < 0:
            raise ValueError("The maximum number of stagnant trials must not be negative.")
        self._max_stagnation_trials = max_stagnation_trials

    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        self._validate_input(trials)
        is_maximize_direction = True if (study_direction == StudyDirection.MAXIMIZE) else False
        trials = [t for t in trials if t.state == TrialState.COMPLETE]
        current_step = len(trials) - 1

        best_step = 0
        for i, trial in enumerate(trials):
            best_value = trials[best_step].value
            current_value = trial.value
            assert best_value is not None
            assert current_value is not None
            if is_maximize_direction and (best_value < current_value):
                best_step = i
            elif (not is_maximize_direction) and (best_value > current_value):
                best_step = i

        return self._max_stagnation_trials - (current_step - best_step)

    @classmethod
    def _validate_input(
        cls,
        trials: List[FrozenTrial],
    ) -> None:
        if len([t for t in trials if t.state == TrialState.COMPLETE]) == 0:
            raise ValueError(
                "Because no trial has been completed yet, the improvement cannot be evaluated."
            )
