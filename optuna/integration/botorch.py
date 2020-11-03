# Implement default optimize_func.
# Allow injecting/overriding from/to uniform.
# Support log, and step. At least log for uniform.
# Implement reseed random.
from collections import OrderedDict
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy
import torch

from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    import botorch
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
            ["torch.Tensor", "torch.Tensor", "torch.Tensor"], List[float]
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
        # TODO(hvy): Warp, unwarp.

        if len(search_space) == 0:
            return {}

        trials = _get_trials(study)
        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        botorch_inputs = _to_botorch_inputs(trials, study.direction, search_space)
        botorch_outputs = self._optimize_func(*botorch_inputs)
        params = _from_botorch_outputs(botorch_outputs, search_space)

        return params

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


def _get_trials(study: Study) -> List[FrozenTrial]:
    # TODO(hvy): Consider pruned trials.
    complete_trials = []
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.COMPLETE:
            complete_trials.append(t)
    return complete_trials


def _to_botorch_inputs(
    trials: List[FrozenTrial], direction: StudyDirection, search_space: OrderedDict
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    # TODO(hvy): Remove debug assertion.
    assert isinstance(search_space, OrderedDict)

    n_trials = len(trials)
    n_params = len(search_space)

    # Allocate memory once with NumPy and convert to torch tensors with zero-copy.
    train_x = numpy.empty((n_trials, n_params), dtype=numpy.float64)
    train_obj = numpy.empty((n_trials, 1), dtype=numpy.float64)
    bounds = numpy.empty((2, n_params), dtype=numpy.float64)

    names = list(search_space.keys())
    for i, trial in enumerate(trials):
        for j, name in enumerate(names):
            train_x[i, j] = trial.params[name]

        # BoTorch assumes maximization, so flip the sign if needed.
        if direction == StudyDirection.MINIMIZE:
            value_to_maximize = -trial.value
        elif direction == StudyDirection.MAXIMIZE:
            value_to_maximize = trial.value
        else:
            assert False
        train_obj[i, 0] = value_to_maximize

    for i, distribution in enumerate(search_space.values()):
        if isinstance(distribution, UniformDistribution):
            bounds[0, i] = distribution.low
            bounds[1, i] = distribution.high
        elif isinstance(distribution, LogUniformDistribution):
            bounds[0, i] = distribution.low
            bounds[1, i] = distribution.high
        else:
            # TODO(hvy): Support integer and categorical parameters similar to how Ax does it on
            # top of BoTorch. https://github.com/pytorch/botorch/issues/177
            assert False, "Distributions besides UniformDistribution are not yet supported."

    # Zero-copy cast.
    train_x = torch.from_numpy(train_x)
    train_obj = torch.from_numpy(train_obj)
    bounds = torch.from_numpy(bounds)

    return train_x, train_obj, bounds


def _to_botorch_param(optuna_param, distribution):
    if isinstance(distribution, UniformDistribution):
        return optuna_param
    elif isinstance(distribution, LogUniformDistribution):
        return math.log(optuna_param)
    elif isinstance(distribution, DiscreteUniformDistribution):
        return optuna_param
    else:
        # TODO(hvy): Support integer and categorical parameters similar to how Ax does it on
        # top of BoTorch. https://github.com/pytorch/botorch/issues/177
        assert False, f"{distribution} is not supported."


def _from_botorch_param(botorch_param, distribution):
    if isinstance(distribution, UniformDistribution):
        return botorch_param
    elif isinstance(distribution, LogUniformDistribution):
        return math.exp(botorch_param)
    elif isinstance(distribution, DiscreteUniformDistribution):
        v = numpy.round(botorch_param / distribution.q) * distribution.q + distribution.low
        # v may slightly exceed range due to round-off errors.
        return float(min(max(v, distribution.low), distribution.high))
    else:
        # TODO(hvy): Support integer and categorical parameters similar to how Ax does it on
        # top of BoTorch. https://github.com/pytorch/botorch/issues/177
        assert False, f"{distribution} is not supported."


def _from_botorch_outputs(candidates: "torch.Tensor", search_space: OrderedDict) -> Dict[str, Any]:
    # TODO(hvy): Remove debug assertion.
    assert isinstance(search_space, OrderedDict)

    if not isinstance(candidates, torch.Tensor):
        raise TypeError

    n_params = len(search_space)

    if candidates.ndim == 2:
        if candidates.shape[0] != 1:
            raise ValueError  # Batch optimization is not supported.
        # Batch size is one. Get rid of the batch dimension.
        candidates = candidates.squeeze(0)
    if candidates.ndim != 1:
        raise ValueError
    if candidates.shape[0] != n_params:
        raise ValueError

    params = {}
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    # TODO(hvy): Continue here to assert the distribution to uniform.
    for (name, distribution), param in zip(search_space.items(), candidates):
        print("name", name, param)
        # TODO(hvy): Unwarp.
        params[name] = param

    return params
