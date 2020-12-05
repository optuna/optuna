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


@experimental("2.4.0")
def ucb_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Upper Confidence Bound (UCB).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization without objective constraints.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Objectives. A ``torch.Tensor`` of shape ``(n_trials, n_objectives)``. ``n_trials``
            is identical to that of ``train_x``. ``n_objectives`` is the number of objectives.
            Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`. In particular for
            :func:`~optuna.integration.botorch.ucb_candidates_func`, it should always be
            :obj:`None` since UCB does not support constrained optimization.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(n_params, 2)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second column correspond to the
            lower and upper bounds for each parameter respectively.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

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


@experimental("2.4.0")
def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization with objective constraints.

    .. seealso::
        :func:`~optuna.integration.botorch.ucb_candidates_func` for argument and return value
        descriptions.
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
    for i in range(n_contraints):
        constraints.append(lambda Z, i=i: Z[..., -n_contraints + i])
    constrained_objective = ConstrainedMCObjective(
        objective=lambda Z: Z[..., 0],
        constraints=constraints,
    )

    acqf = qExpectedImprovement(
        model=model,
        best_f=(train_obj * (train_con <= 0)).max(),
        sampler=SobolQMCNormalSampler(num_samples=256),
        objective=constrained_objective,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


@experimental("2.4.0")
def qehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Expected Hypervolume Improvement (qEHVI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization with or without objective constraints.

    .. seealso::
        :func:`~optuna.integration.botorch.ucb_candidates_func` for argument and return value
        descriptions.
    """

    n_outcomes = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        partitioning_y = train_obj[is_feas]

        constraints = []
        n_contraints = train_con.size(1)

        for i in range(n_contraints):
            constraints.append(lambda Z, i=i: Z[..., -n_contraints + i])
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
        sampler=SobolQMCNormalSampler(num_samples=256),
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
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna/blob/master/examples/multi_objective/
        botorch_simple.py>`_ how to use the sampler.

    .. seealso::
        See the `BoTorch <https://botorch.org/>`_ homepage for details and for how to implement
        your own ``candidates_func``.

    .. note::
        An instance of this sampler *should be not used with different studies* when used with
        constraints. Instead, a new instance should be created for each new study. The reason for
        this is that the sampler is stateful keeping all the computed constraints.

    Args:
        candidates_func:
            An optional function that suggests the next candidates. It must take the training
            data, the objectives, the constraints, the search space bounds and return the next
            candidates. The arguments are of type ``torch.Tensor``. The return value must be a
            ``torch.Tensor``. However, if ``constraints_func`` is omitted, constraints will be
            :obj:`None`.

            If omitted, is determined automatically based on the number of objectives and
            constraints.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.ucb_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.multi_objective.study.MultiObjectiveStudy`, a
            :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` and return the
            constraints. The return value must be a sequence of :obj:`float` s. A value strictly
            larger than 0 means that a constraints is violated. A value equal to or smaller than 0
            is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion if ``candidates_func`` is omitted.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
    """

    def __init__(
        self,
        candidates_func: Callable[
            [
                "torch.Tensor",
                "torch.Tensor",
                Optional["torch.Tensor"],
                "torch.Tensor",
            ],
            "torch.Tensor",
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
        n_startup_trials: int = 10,
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
