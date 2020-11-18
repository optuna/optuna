from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import Optional

import numpy

from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch


def optimize_func(
    train_x: torch.Tensor, train_obj: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    # Initialize and fit GP.
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Optimize acquisition function.
    acqf = UpperConfidenceBound(model, beta=0.1)
    candidates, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=40)

    return candidates


@experimental("2.4.0")
class BoTorchSampler(BaseSampler):
    def __init__(
        self,
        optimize_func: Callable[
            ["torch.Tensor", "torch.Tensor", "torch.Tensor"], torch.Tensor
        ] = optimize_func,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
    ):
        _imports.check()

        self._optimize_func = optimize_func
        self._independent_sampler = independent_sampler or RandomSampler()
        self._n_startup_trials = n_startup_trials
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self._search_space.calculate(study, ordered_dict=True)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, float]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        values = numpy.asarray([t.value for t in trials], dtype=numpy.float64)

        trans = _SearchSpaceTransform(search_space)
        bounds = trans.bounds
        params = numpy.empty((n_trials, bounds.shape[0]), dtype=numpy.float64)
        for i, t in enumerate(trials):
            params[i] = trans.transform(t.params)

        values = torch.from_numpy(values)
        params = torch.from_numpy(params)
        bounds = torch.from_numpy(bounds)

        values.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        # BoTorch always assumes maximization.
        if study.direction == StudyDirection.MINIMIZE:
            values *= -1

        candidates = self._optimize_func(params, values, bounds)

        # TODO(hvy): Clean up validation.
        if not isinstance(candidates, torch.Tensor):
            raise TypeError
        if candidates.dim() == 2:
            if candidates.shape[0] != 1:
                raise ValueError  # Batch optimization is not supported.
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError
        if candidates.shape[0] != bounds.shape[1]:
            raise ValueError

        candidates = candidates.numpy()

        return trans.untransform(candidates)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        # TODO(hvy): Implement.
        raise NotImplementedError
