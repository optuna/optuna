from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy

from optuna import multi_objective
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.multi_objective.samplers._random import RandomMultiObjectiveSampler
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.samplers import IntersectionSearchSpace
from optuna.study import StudyDirection
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition import UpperConfidenceBound
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
    from botorch.utils.transforms import unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch


def ucb_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Upper Confidence Bound (UCB).

    Single-objective, without objective constraints.
    """

    if train_con is not None:
        warnings.warn("Constraints are given but will be ignored.")

    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    acqf = UpperConfidenceBound(model, beta=0.1)

    candidates, _ = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=40,
    )

    return candidates


def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Expected Improvement (qEI).

    Single-objective, with objective constraints.
    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is None:
        raise ValueError("Constraints must be used with qEI.")

    train_y = torch.cat([train_obj, train_con], dim=-1)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    constraints = []
    n_contraints = train_con.size(1)
    for i in range(1, n_contraints + 1):
        constraints.append(lambda Z, i=i: Z[..., -i])
    constrained_objective = ConstrainedMCObjective(
        objective=lambda Z: Z[..., 0],
        constraints=constraints,
    )

    acqf = qExpectedImprovement(
        model=model,
        best_f=(train_obj * (train_con <= 0)).max(),
        sampler=SobolQMCNormalSampler(num_samples=128),
        objective=constrained_objective,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def qehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Expected Hypervolume Improvement (qEHVI).

    Multi-objective, with and without objective constraints.
    """

    n_outcomes = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        partitioning_y = train_obj[is_feas]

        constraints = []
        n_contraints = train_con.size(1)
        for i in range(1, n_contraints + 1):
            constraints.append(lambda Z, i=i: Z[..., -i])
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_outcomes))),
            "constraints": constraints,
        }
    else:
        train_y = train_obj

        partitioning_y = train_obj

        additional_qehvi_kwargs = {}

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    partitioning = NondominatedPartitioning(num_outcomes=n_outcomes, Y=partitioning_y)

    ref_point = train_obj.amin(dim=0) - 1e-8
    ref_point_list = ref_point.tolist()

    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(num_samples=128),
        **additional_qehvi_kwargs,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def _get_default_candidates_func(
    n_objectives: int, n_objective_constraints: int
) -> Callable[
    [
        "torch.Tensor",
        "torch.Tensor",
        Optional["torch.Tensor"],
        "torch.Tensor",
    ],
    "torch.Tensor",
]:
    assert n_objectives >= 1
    assert n_objective_constraints >= 0

    if n_objectives == 1:
        if n_objective_constraints == 0:
            return ucb_candidates_func
        else:
            return qei_candidates_func
    else:
        return qehvi_candidates_func


# TODO(hvy): Allow utilizing GPUs via some parameter, not having to rewrite the callback
# functions.
@experimental("2.4.0")
class BoTorchSampler(BaseMultiObjectiveSampler):
    def __init__(
        self,
        candidates_func: Callable[
            [
                "torch.Tensor",
                "torch.Tensor",
                Optional["torch.Tensor"],
                "torch.Tensor",
            ],
            torch.Tensor,
        ] = None,
        constraints_func: Optional[
            Callable[
                [
                    "MultiObjectiveStudy",
                    "FrozenMultiObjectiveTrial",
                ],
                Sequence[float],
            ]
        ] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseMultiObjectiveSampler] = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._independent_sampler = independent_sampler or RandomMultiObjectiveSampler()
        self._n_startup_trials = n_startup_trials

        self._trial_constraints: Dict[int, Tuple[float, ...]] = {}
        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemortyStorage` is used
            # because `InMemortyStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        return self._search_space.calculate(study, ordered_dict=True)  # type: ignore

    def _update_trial_constraints(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trials: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    ) -> None:
        # Since trial constraints are computed on each worker, constraints should be computed
        # deterministically.

        assert self._constraints_func is not None

        for trial in trials:
            number = trial.number
            if number not in self._trial_constraints:
                constraints = self._constraints_func(study, trial)
                if not isinstance(constraints, (tuple, list)):
                    raise TypeError("Constraints must be a tuple or list.")
                constraints = tuple(constraints)
                self._trial_constraints[number] = constraints

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

        if self._constraints_func is not None:
            self._update_trial_constraints(study, trials)

        trans = _SearchSpaceTransform(search_space)

        values = numpy.empty((n_trials, study.n_objectives), dtype=numpy.float64)
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        if self._constraints_func is not None:
            n_objective_constraints = len(next(iter(self._trial_constraints.values())))
            con = numpy.empty((n_trials, n_objective_constraints), dtype=numpy.float64)
        else:
            n_objective_constraints = 0
            con = None
        bounds = trans.bounds

        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

            if con is not None:
                con[trial_idx] = self._trial_constraints[trial_idx]

        values = torch.from_numpy(values)
        params = torch.from_numpy(params)
        if con is not None:
            con = torch.from_numpy(con)
        bounds = torch.from_numpy(bounds)

        if values.dim() == 1:
            values.unsqueeze_(-1)
        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=study.n_objectives, n_objective_constraints=n_objective_constraints
            )
        candidates = self._candidates_func(params, values, con, bounds)

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

        params = trans.untransform(candidates)

        # Exclude upper bounds for parameters that should have their upper bounds excluded.
        # TODO(hvy): Remove this exclusion logic when it is handled by the data transformer.
        for name, param in params.items():
            if isinstance(search_space[name], (UniformDistribution, LogUniformDistribution)):
                params[name] = min(params[name], search_space[name].high - 1e-8)

        return params

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
        self._independent_sampler.reseed_rng()
