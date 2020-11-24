from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Optional

import numpy

from optuna import multi_objective
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.multi_objective.samplers._random import RandomMultiObjectiveSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.study import StudyDirection
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch


# def optimize_func(
#     train_x: torch.Tensor, train_obj: torch.Tensor, bounds: torch.Tensor
# ) -> torch.Tensor:
#     # Initialize and fit GP.
#     model = SingleTaskGP(train_x, train_obj)
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     fit_gpytorch_model(mll)
#
#     # Optimize acquisition function.
#     acqf = UpperConfidenceBound(model, beta=0.1)
#     candidates, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=40)
#
#     return candidates


@experimental("2.4.0")
class BoTorchSampler(BaseMultiObjectiveSampler):
    def __init__(
        self,
        optimize_func: Callable[
            ["torch.Tensor", "torch.Tensor", "torch.Tensor"], torch.Tensor
        ] = None,
        constraints: Optional[Callable[["Study", "FrozenTrial"], Sequence[float]]] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseMultiObjectiveSampler] = None,
    ):
        _imports.check()

        if optimize_func is None:
            raise NotImplementedError

        self._optimize_func = optimize_func
        self._independent_sampler = independent_sampler or RandomMultiObjectiveSampler()
        self._n_startup_trials = n_startup_trials
        self._search_space = IntersectionSearchSpace()
        self._trial_constraints: Dict[int, Tuple[float, ...]] = {}
        self._constraints = constraints
        self._study_id: Optional[int] = None

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        return self._search_space.calculate(study, ordered_dict=True)  # type: ignore

    def _update_trial_constraints(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trials: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    ) -> None:
        # Since trial constraints are computed on each worker, constraints should be computed
        # deterministically.

        assert self._constraints is not None

        for trial in trials:
            number = trial.number
            if number not in self._trial_constraints:
                self._trial_constraints[number] = self._constraints(study, trial)

    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        if self._constraints is not None:
            self._update_trial_constraints(study, trials)

        trans = _SearchSpaceTransform(search_space)

        values = numpy.empty((n_trials, study.n_objectives), dtype=numpy.float64)
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        if self._trial_constraints is not None:
            n_contraints = len(next(iter(self._trial_constraints.values())))
            cons = numpy.empty((n_trials, n_contraints), dtype=numpy.float64)
        else:
            cons = None
        bounds = trans.bounds

        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

            if cons is not None:
                cons[trial_idx] = self._trial_constraints[trial_idx]

        values = torch.from_numpy(values)
        params = torch.from_numpy(params)
        if cons is not None:
            cons = torch.from_numpy(cons)
        bounds = torch.from_numpy(bounds)

        if values.dim() == 1:
            values.unsqueeze_(-1)
        if cons.dim() == 1:
            cons.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        candidates = self._optimize_func(params, values, cons, bounds)

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
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        # TODO(hvy): Implement.
        raise NotImplementedError
