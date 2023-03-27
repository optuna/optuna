import abc
from typing import Dict
from typing import List

import numpy as np

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import StudyDirection
from optuna.terminator import _distribution_is_log
from optuna.terminator._search_space.intersection import IntersectionSearchSpace
from optuna.terminator.improvement._preprocessing import BasePreprocessing
from optuna.terminator.improvement._preprocessing import OneToHot
from optuna.terminator.improvement._preprocessing import PreprocessingPipeline
from optuna.terminator.improvement._preprocessing import SelectTopTrials
from optuna.terminator.improvement._preprocessing import ToMinimize
from optuna.terminator.improvement._preprocessing import UnscaleLog
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition.analytic import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    import gpytorch
    import torch

DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


@experimental_class("3.2.0")
class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        pass


@experimental_class("3.2.0")
class RegretBoundEvaluator(BaseImprovementEvaluator):
    def __init__(
        self,
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        min_lcb_n_additional_samples: int = 2000,
    ) -> None:
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
            OneToHot(),
        ]

        return PreprocessingPipeline(processes)

    def evaluate(
        self,
        trials: List[FrozenTrial],
        study_direction: StudyDirection,
    ) -> float:
        search_space = IntersectionSearchSpace().calculate(trials)
        self._validate_input(trials, search_space)

        fit_trials = self.get_preprocessing().apply(trials, study_direction)

        x, bounds = _convert_trials_to_tensors(trials)

        y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)
        y = torch.unsqueeze(y, 1)

        gp = SingleTaskGP(
            x,
            y,
            input_transform=Normalize(d=x.shape[1], bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

        fit_gpytorch_model(mll)

        beta = _get_beta(n_params=len(search_space), n_trials=len(fit_trials))
        neg_lcb_func = UpperConfidenceBound(gp, beta=beta, maximize=False)
        ucb_func = UpperConfidenceBound(gp, beta=beta, maximize=True)

        with gpytorch.settings.fast_pred_var():  # type: ignore[no-untyped-call]
            min_ucb = torch.min(-ucb_func(x[:, None, :])).item()

            x_opt, lcb = optimize_acqf(
                neg_lcb_func, bounds=bounds, q=1, num_restarts=10, raw_samples=512, sequential=True
            )

            min_lcb = -lcb.item()

        return min_ucb - min_lcb

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


def _convert_trials_to_tensors(trials: list[FrozenTrial]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of FrozenTrial objects to tensors inputs and bounds.

    This function assumes the following condition for input trials:
    - any categorical param is converted to a float or int one;
    - log is unscaled for any float/int distribution;
    - the state is COMPLETE for any trial;
    - direction is MINIMIZE for any trial.
    """
    search_space = IntersectionSearchSpace().calculate(trials)
    sorted_params = sorted(search_space.keys())

    x = []
    for trial in trials:
        assert trial.state == TrialState.COMPLETE
        x_row = []
        for param in sorted_params:
            distribution = search_space[param]

            assert not _distribution_is_log(distribution)
            assert not isinstance(distribution, CategoricalDistribution)

            param_value = float(trial.params[param])
            x_row.append(param_value)

        x.append(x_row)

    min_bounds = []
    max_bounds = []
    for param, distribution in search_space.items():
        assert isinstance(distribution, (FloatDistribution, IntDistribution))
        min_bounds.append(distribution.low)
        max_bounds.append(distribution.high)
    bounds = [min_bounds, max_bounds]

    return torch.tensor(x, dtype=torch.float64), torch.tensor(bounds, dtype=torch.float64)


def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
    beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

    # The following div is according to the original paper: "We then further scale it down
    # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
    beta /= 5

    return beta
