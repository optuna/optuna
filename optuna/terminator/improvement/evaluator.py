from __future__ import annotations

import abc
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna._gp import gp
from optuna._gp import optim
import optuna._gp.acqf as acqf
from optuna._gp.prior import default_log_prior
from optuna._gp.prior import DEFAULT_MINIMUM_NOISE_VAR
from optuna._gp.search_space import get_search_space_and_normalized_params
from optuna._gp.search_space import ScaleType
from optuna.distributions import BaseDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import intersection_search_space
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
    beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

    # The following div is according to the original paper: "We then further scale it down
    # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
    beta /= 5

    return beta


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
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        seed: int | None = None,
    ) -> None:
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials
        self._log_prior = default_log_prior
        self._minimum_noise = DEFAULT_MINIMUM_NOISE_VAR
        self._optimize_n_samples = 2048
        self._rng = LazyRandomState(seed)

    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        optuna_search_space = intersection_search_space(trials)
        self._validate_input(trials, optuna_search_space)

        complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        # _gp module assumes that optimization direction is maximization
        sign = -1 if study_direction == StudyDirection.MINIMIZE else 1
        values = np.array([t.value for t in complete_trials]) * sign
        gp_search_space, normalized_params = get_search_space_and_normalized_params(
            complete_trials, optuna_search_space
        )

        top_n = int(len(trials) * self._top_trials_ratio)
        top_n = max(top_n, self._min_n_trials)
        top_n = min(top_n, len(trials))
        indices = np.argsort(-values)[:top_n]

        top_n_values = values[indices]
        top_n_values_mean = top_n_values.mean()
        top_n_values_std = max(1e-10, top_n_values.std())

        standarized_top_n_values = (top_n_values - top_n_values_mean) / top_n_values_std
        normalized_top_n_params = normalized_params[indices]

        kernel_params = gp.fit_kernel_params(
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            is_categorical=(gp_search_space.scale_types == ScaleType.CATEGORICAL),
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            initial_kernel_params=None,
        )

        n_params = len(optuna_search_space)

        # calculate max_ucb
        sqrt_beta = np.sqrt(_get_beta(n_params, top_n))
        ucb_acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.UCB,
            kernel_params=kernel_params,
            search_space=gp_search_space,
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            sqrt_beta=sqrt_beta,
        )
        _, standardized_ucb_value = optim.optimize_acqf_sample(
            ucb_acqf_params,
            n_samples=self._optimize_n_samples,
            seed=self._rng.rng.randint(np.iinfo(np.int32).max),
        )
        with torch.no_grad():  # type: ignore
            standardized_ucb_value = max(
                standardized_ucb_value,
                acqf.eval_acqf(ucb_acqf_params, torch.from_numpy(normalized_top_n_params))
                .max()
                .item(),
            )
        ucb_value = standardized_ucb_value * top_n_values_std + top_n_values_mean

        lcb_acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.UCB,
            kernel_params=kernel_params,
            search_space=gp_search_space,
            X=normalized_top_n_params,
            Y=standarized_top_n_values,
            sqrt_beta=-sqrt_beta,
        )
        with torch.no_grad():  # type: ignore
            standardized_lcb_value = (
                acqf.eval_acqf(lcb_acqf_params, torch.from_numpy(normalized_top_n_params))
                .max()
                .item()
            )
        lcb_value = standardized_lcb_value * top_n_values_std + top_n_values_mean

        regret_bound = ucb_value - lcb_value

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
