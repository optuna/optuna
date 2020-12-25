from collections import OrderedDict
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
import warnings

import numpy

from optuna import logging
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
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
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import sample_simplex
    from botorch.utils.transforms import normalize
    from botorch.utils.transforms import unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch


_logger = logging.get_logger(__name__)


@experimental("2.4.0")
def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(n_params, 2)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second column correspond to the
            lower and upper bounds for each parameter respectively.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        if train_obj_feas.numel() == 0:
            # TODO(hvy): Do not use 0 as the best observation.
            _logger.warning(
                "No objective values are feasible. Using 0 as the best objective in qEI."
            )
            best_f = torch.zeros(())
        else:
            best_f = train_obj_feas.max()

        constraints = []
        n_contraints = train_con.size(1)
        for i in range(n_contraints):
            constraints.append(lambda Z, i=i: Z[..., -n_contraints + i])
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=constraints,
        )
    else:
        train_y = train_obj

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(num_samples=256),
        objective=objective,
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
    """Quasi MC-based batch Expected Hypervolume Improvement (qEHVI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        constraints = []
        n_contraints = train_con.size(1)

        for i in range(n_contraints):
            constraints.append(lambda Z, i=i: Z[..., -n_contraints + i])
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": constraints,
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0
    partitioning = NondominatedPartitioning(
        num_outcomes=n_objectives, Y=train_obj_feas, alpha=alpha
    )

    ref_point = train_obj.min(dim=0).values - 1e-8
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


@experimental("2.4.0")
def qparego_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Quasi MC-based extended ParEGO (qParEGO) for constrained multi-objective optimization.

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is larger than three.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    weights = sample_simplex(n_objectives).squeeze()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        constraints = []
        n_contraints = train_con.size(1)

        for i in range(n_contraints):
            constraints.append(lambda Z, i=i: Z[..., -n_contraints + i])

        objective = ConstrainedMCObjective(
            objective=lambda Z: scalarization(Z[..., :n_objectives]),
            constraints=constraints,
        )
    else:
        train_y = train_obj

        objective = GenericMCObjective(scalarization)

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=objective(train_y).max(),
        sampler=SobolQMCNormalSampler(num_samples=256),
        objective=objective,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def _get_default_candidates_func(
    n_objectives: int,
) -> Callable[
    [
        "torch.Tensor",
        "torch.Tensor",
        Optional["torch.Tensor"],
        "torch.Tensor",
    ],
    "torch.Tensor",
]:
    if n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    else:
        return qei_candidates_func


# TODO(hvy): Allow utilizing GPUs via some parameter, not having to rewrite the callback
# functions.
@experimental("2.4.0")
class BoTorchSampler(BaseSampler):
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna/blob/master/examples/
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
            :obj:`None`. For any constraints that failed to compute, the tensor will contain
            NaN.

            If omitted, is determined automatically based on the number of objectives. If the
            number of objectives is one, Quasi MC-based batch Expected Improvement (qEI) is used.
            If the number of objectives is larger than one but smaller than four, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number
            of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.

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
        *,
        candidates_func: Callable[
            [
                "torch.Tensor",
                "torch.Tensor",
                Optional["torch.Tensor"],
                "torch.Tensor",
            ],
            "torch.Tensor",
        ] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        n_startup_trials: int = 10,
        independent_sampler: Optional[BaseSampler] = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._independent_sampler = independent_sampler or RandomSampler()
        self._n_startup_trials = n_startup_trials

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemortyStorage` is used
            # because `InMemortyStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        return self._search_space.calculate(study, ordered_dict=True)  # type: ignore

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)

        n_objectives = len(study.directions)
        values = numpy.empty((n_trials, n_objectives), dtype=numpy.float64)
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
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

            if self._constraints_func is not None:
                constraints = study._storage.get_trial_system_attrs(trial._trial_id).get(
                    "botorch:constraints"
                )
                if constraints is not None:
                    n_constraints = len(constraints)

                    if con is None:
                        con = numpy.full((n_trials, n_constraints), numpy.nan, dtype=numpy.float64)
                    elif n_constraints != con.shape[1]:
                        raise RuntimeError(
                            f"Expected {con.shape[1]} constraints but received {n_constraints}."
                        )

                    con[trial_idx] = constraints

        if self._constraints_func is not None:
            if con is None:
                warnings.warn(
                    "`constraints_func` was given but no call to it correctly computed "
                    "constraints. Constraints passed to `candidates_func` will be `None`."
                )
            elif numpy.isnan(con).any():
                warnings.warn(
                    "`constraints_func` was given but some calls to it did not correctly compute "
                    "constraints. Constraints passed to `candidates_func` will contain NaN."
                )

        values = torch.from_numpy(values)
        params = torch.from_numpy(params)
        if con is not None:
            con = torch.from_numpy(con)
        bounds = torch.from_numpy(bounds)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(n_objectives=n_objectives)
        candidates = self._candidates_func(params, values, con, bounds)

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        candidates = candidates.numpy()

        params = trans.untransform(candidates)

        # Exclude upper bounds for parameters that should have their upper bounds excluded.
        # TODO(hvy): Remove this exclusion logic when it is handled by the data transformer.
        for name, param in params.items():
            distribution = search_space[name]
            if isinstance(distribution, UniformDistribution):
                params[name] = min(params[name], search_space[name].high - 1e-8)
            elif isinstance(distribution, LogUniformDistribution):
                params[name] = min(
                    params[name], math.exp(math.log(search_space[name].high) - 1e-8)
                )

        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            constraints = None

            try:
                con = self._constraints_func(trial)
                if not isinstance(con, (tuple, list)):
                    warnings.warn(
                        f"Constraints should be a sequence of floats but got {constraints}."
                    )
                constraints = tuple(con)
            except Exception:
                raise
            finally:
                assert constraints is None or isinstance(constraints, tuple)

                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    "botorch:constraints",
                    constraints,
                )
