from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import numpy
from packaging import version

from optuna import logging
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    from botorch.acquisition.multi_objective import monte_carlo
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    import botorch.version

    if version.parse(botorch.version.version) < version.parse("0.8.0"):
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            return SobolQMCNormalSampler(num_samples)

    else:
        from botorch.fit import fit_gpytorch_mll

        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            return SobolQMCNormalSampler(torch.Size((num_samples,)))

    from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import manual_seed
    from botorch.utils.sampling import sample_simplex
    from botorch.utils.transforms import normalize
    from botorch.utils.transforms import unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch


_logger = logging.get_logger(__name__)

with try_import() as _imports_logei:
    from botorch.acquisition.analytic import LogExpectedImprovement


@experimental_func("3.3.0")
def logei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """Log Expected Improvement (LogEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization for non-constrained problems.

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
            Objective constraints. This option is not supported in ``logei_candidates_func`` and
            must be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.
        pending_x:
            Pending parameter configurations. A ``torch.Tensor`` of shape
            ``(n_pending, n_params)``. ``n_pending`` is the number of the trials which are already
            suggested all their parameters but have not completed their evaluation, and
            ``n_params`` is identical to that of ``train_x``.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    # We need botorch >=0.8.1 for LogExpectedImprovement.
    if not _imports_logei.is_successful():
        raise ImportError(
            "logei_candidates_func requires botorch >=0.8.1. "
            "Please upgrade botorch or use qei_candidates_func as candidates_func instead."
        )

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with logEI.")
    if train_con is not None:
        raise ValueError(
            "Constraint is not supported with logei_candidates_func. "
            + "Please use qei_candidates_func instead."
        )
    else:
        train_y = train_obj
        best_f = train_obj.max()

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = LogExpectedImprovement(
        model=model,
        best_f=best_f,
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


@experimental_func("2.4.0")
def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization for constrained problems.

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
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.
        pending_x:
            Pending parameter configurations. A ``torch.Tensor`` of shape
            ``(n_pending, n_params)``. ``n_pending`` is the number of the trials which are already
            suggested all their parameters but have not completed their evaluation, and
            ``n_params`` is identical to that of ``train_x``.
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

        n_constraints = train_con.size(1)
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=[
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        )
    else:
        train_y = train_obj

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=_get_sobol_qmc_normal_sampler(256),
        objective=objective,
        X_pending=pending_x,
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


@experimental_func("3.3.0")
def qnei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """Quasi MC-based batch Noisy Expected Improvement (qNEI).

    This function may perform better than qEI (`qei_candidates_func`) when
    the evaluated values of objective function are noisy.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """
    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qNEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=[
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        )
    else:
        train_y = train_obj

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=_get_sobol_qmc_normal_sampler(256),
        objective=objective,
        X_pending=pending_x,
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


@experimental_func("2.4.0")
def qehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
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

        n_constraints = train_con.size(1)
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": [
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj_feas, alpha=alpha)

    ref_point_list = ref_point.tolist()

    acqf = monte_carlo.qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
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


@experimental_func("3.1.0")
def qnehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """Quasi MC-based batch Noisy Expected Hypervolume Improvement (qNEHVI).

    According to Botorch/Ax documentation,
    this function may perform better than qEHVI (`qehvi_candidates_func`).
    (cf. https://botorch.org/tutorials/constrained_multi_objective_bo )

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        additional_qnehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": [
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        }
    else:
        train_y = train_obj

        additional_qnehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    ref_point_list = ref_point.tolist()

    # prune_baseline=True is generally recommended by the documentation of BoTorch.
    # cf. https://botorch.org/api/acquisition.html (accessed on 2022/11/18)
    acqf = monte_carlo.qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        alpha=alpha,
        prune_baseline=True,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qnehvi_kwargs,
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


@experimental_func("2.4.0")
def qparego_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"],
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
        n_constraints = train_con.size(1)
        objective = ConstrainedMCObjective(
            objective=lambda Z: scalarization(Z[..., :n_objectives]),
            constraints=[
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        )
    else:
        train_y = train_obj

        objective = GenericMCObjective(scalarization)

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=objective(train_y).max(),
        sampler=_get_sobol_qmc_normal_sampler(256),
        objective=objective,
        X_pending=pending_x,
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
    has_constraint: bool,
    consider_running_trials: bool,
) -> Callable[
    [
        "torch.Tensor",
        "torch.Tensor",
        Optional["torch.Tensor"],
        "torch.Tensor",
        Optional["torch.Tensor"],
    ],
    "torch.Tensor",
]:
    if n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    elif has_constraint or consider_running_trials:
        return qei_candidates_func
    else:
        return logei_candidates_func


@experimental_class("2.4.0")
class BoTorchSampler(BaseSampler):
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna-examples/blob/main/multi_objective/
        botorch_simple.py>`_ how to use the sampler.

    .. seealso::
        See the `BoTorch <https://botorch.org/>`_ homepage for details and for how to implement
        your own ``candidates_func``.

    .. note::
        An instance of this sampler *should not be used with different studies* when used with
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

            If omitted, it is determined automatically based on the number of objectives and
            whether a constraint is specified. If the
            number of objectives is one and no constraint is specified, log-Expected Improvement
            is used. If constraints are specified, quasi MC-based batch Expected Improvement
            (qEI) is used.
            If the number of objectives is either two or three, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number
            of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        consider_running_trials:
            If True, the acquisition function takes into consideration the running parameters
            whose evaluation has not completed. Enabling this option is considered to improve the
            performance of parallel optimization.

            .. note::
                Added in v3.2.0 as an experimental argument.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
        seed:
            Seed for random number generator.
        device:
            A ``torch.device`` to store input and output data of BoTorch. Please set a CUDA device
            if you fasten sampling.
    """

    def __init__(
        self,
        *,
        candidates_func: Optional[
            Callable[
                [
                    "torch.Tensor",
                    "torch.Tensor",
                    Optional["torch.Tensor"],
                    "torch.Tensor",
                    Optional["torch.Tensor"],
                ],
                "torch.Tensor",
            ]
        ] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        device: Optional["torch.device"] = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._consider_running_trials = consider_running_trials
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, dict)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        running_trials = [
            t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)) if t != trial
        ]
        trials = completed_trials + running_trials

        n_trials = len(trials)
        n_completed_trials = len(completed_trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty(
            (n_trials, n_objectives), dtype=numpy.float64
        )
        params: Union[numpy.ndarray, torch.Tensor]
        con: Optional[Union[numpy.ndarray, torch.Tensor]] = None
        bounds: Union[numpy.ndarray, torch.Tensor] = trans.bounds
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        for trial_idx, trial in enumerate(trials):
            if trial.state == TrialState.COMPLETE:
                params[trial_idx] = trans.transform(trial.params)
                assert len(study.directions) == len(trial.values)
                for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                    assert value is not None
                    if (
                        direction == StudyDirection.MINIMIZE
                    ):  # BoTorch always assumes maximization.
                        value *= -1
                    values[trial_idx, obj_idx] = value
                if self._constraints_func is not None:
                    constraints = study._storage.get_trial_system_attrs(trial._trial_id).get(
                        _CONSTRAINTS_KEY
                    )
                    if constraints is not None:
                        n_constraints = len(constraints)

                        if con is None:
                            con = numpy.full(
                                (n_completed_trials, n_constraints), numpy.nan, dtype=numpy.float64
                            )
                        elif n_constraints != con.shape[1]:
                            raise RuntimeError(
                                f"Expected {con.shape[1]} constraints "
                                f"but received {n_constraints}."
                            )
                        con[trial_idx] = constraints
            elif trial.state == TrialState.RUNNING:
                if all(p in trial.params for p in search_space):
                    params[trial_idx] = trans.transform(trial.params)
                else:
                    params[trial_idx] = numpy.nan
            else:
                assert False, "trail.state must be TrialState.COMPLETE or TrialState.RUNNING."

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

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        if con is not None:
            con = torch.from_numpy(con).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=n_objectives,
                has_constraint=con is not None,
                consider_running_trials=self._consider_running_trials,
            )

        completed_values = values[:n_completed_trials]
        completed_params = params[:n_completed_trials]
        if self._consider_running_trials:
            running_params = params[n_completed_trials:]
            running_params = running_params[~torch.isnan(running_params).any(dim=1)]
        else:
            running_params = None

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            candidates = self._candidates_func(
                completed_params, completed_values, con, bounds, running_params
            )
            if self._seed is not None:
                self._seed += 1

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

        return trans.untransform(candidates.cpu().numpy())

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
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)
