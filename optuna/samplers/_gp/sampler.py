from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import warn_experimental_argument
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.study._multi_objective import _is_pareto_front
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch

    import optuna._gp.acqf as acqf
    import optuna._gp.gp as gp
    import optuna._gp.optim_mixed as optim_mixed
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    acqf = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")


EPS = 1e-10


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


@experimental_class("3.6.0")
class GPSampler(BaseSampler):
    """Sampler using Gaussian process-based Bayesian optimization.

    This sampler fits a Gaussian process (GP) to the objective function and optimizes
    the acquisition function to suggest the next parameters.

    The current implementation uses Matern kernel with nu=2.5 (twice differentiable) with automatic
    relevance determination (ARD) for the length scale of each parameter.
    The hyperparameters of the kernel are obtained by maximizing the marginal log-likelihood of the
    hyperparameters given the past trials.
    To prevent overfitting, Gamma prior is introduced for kernel scale and noise variance and
    a hand-crafted prior is introduced for inverse squared lengthscales.

    As an acquisition function, we use:

    - log expected improvement (logEI) for single-objective optimization,
    - log expected hypervolume improvement (logEHVI) for Multi-objective optimization, and
    - the summation of logEI and the logarithm of the feasible probability with the independent
      assumption of each constraint for (black-box inequality) constrained optimization.

    For further information about these acquisition functions, please refer to the following
    papers:

    - `Unexpected Improvements to Expected Improvement for Bayesian Optimization
      <https://arxiv.org/abs/2310.20708>`__
    - `Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian
      Optimization <https://arxiv.org/abs/2006.05078>`__
    - `Bayesian Optimization with Inequality Constraints
      <https://proceedings.mlr.press/v32/gardner14.pdf>`__

    The optimization of the acquisition function is performed via:

    1. Collect the best param from the past trials,
    2. Collect ``n_preliminary_samples`` points using Quasi-Monte Carlo (QMC) sampling,
    3. Choose the best point from the collected points,
    4. Choose ``n_local_search - 2`` points from the collected points using the roulette
       selection,
    5. Perform a local search for each chosen point as an initial point, and
    6. Return the point with the best acquisition function value as the next parameter.

    Note that the procedures for non single-objective optimization setups are slightly different
    from the single-objective version described above, but we omit the descriptions for the others
    for brevity.

    The local search iteratively optimizes the acquisition function by repeating:

    1. Gradient ascent using l-BFGS-B for continuous parameters, and
    2. Line search or exhaustive search for each discrete parameter independently.

    The local search is terminated if the routine stops updating the best parameter set or the
    maximum number of iterations is reached.

    We use line search instead of rounding the results from the continuous optimization since EI
    typically yields a high value between one grid and its adjacent grid.

    .. note::
        This sampler requires ``scipy`` and ``torch``.
        You can install these dependencies with ``pip install scipy torch``.

    Args:
        seed:
            Random seed to initialize internal random number generator.
            Defaults to :obj:`None` (a seed is picked randomly).
        independent_sampler:
            Sampler used for initial sampling (for the first ``n_startup_trials`` trials)
            and for conditional parameters. Defaults to :obj:`None`
            (a random sampler with the same ``seed`` is used).
        n_startup_trials:
            Number of initial trials. Defaults to 10.
        deterministic_objective:
            Whether the objective function is deterministic or not.
            If :obj:`True`, the sampler will fix the noise variance of the surrogate model to
            the minimum value (slightly above 0 to ensure numerical stability).
            Defaults to :obj:`False`. Currently, all the objectives will be assume to be
            deterministic if :obj:`True`.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or are pruned, but this behavior is
            subject to change in future releases.
            Currently, the ``constraints_func`` option is not supported for multi-objective
            optimization.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[[gp.KernelParamsTensor], torch.Tensor] = prior.default_log_prior
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        # We cache the kernel parameters for initial values of fitting the next time.
        # TODO(nabenabe): Make the cache lists system_attrs to make GPSampler stateless.
        self._kernel_params_cache_list: list[gp.KernelParamsTensor] | None = None
        self._constraints_kernel_params_cache: list[gp.KernelParamsTensor] | None = None
        self._deterministic = deterministic_objective
        self._constraints_func = constraints_func

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")

        # Control parameters of the acquisition function optimization.
        self._n_preliminary_samples: int = 2048
        # NOTE(nabenabe): ehvi in BoTorchSampler uses 20.
        self._n_local_search = 10
        self._tol = 1e-4

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _optimize_acqf(
        self,
        acqf_params: acqf.AcquisitionFunctionParams,
        best_params: np.ndarray | None,
    ) -> np.ndarray:
        # Advanced users can override this method to change the optimization algorithm.
        # However, we do not make any effort to keep backward compatibility between versions.
        # Particularly, we may remove this function in future refactoring.
        assert best_params is None or len(best_params.shape) == 2
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf_params,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params

    def _get_constraints_acqf_params(
        self,
        constraint_vals: np.ndarray,
        internal_search_space: gp_search_space.SearchSpace,
        normalized_params: np.ndarray,
    ) -> list[acqf.AcquisitionFunctionParams]:
        standardized_constraint_vals, means, stds = _standardize_values(constraint_vals)
        if self._kernel_params_cache_list is not None and len(
            self._kernel_params_cache_list[0].inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            # Clear cache if the search space changes.
            self._constraints_kernel_params_cache = None

        is_categorical = internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        constraints_kernel_params = []
        constraints_acqf_params = []
        for i, (vals, mean, std) in enumerate(zip(standardized_constraint_vals.T, means, stds)):
            cache = (
                self._constraints_kernel_params_cache[i]
                if self._constraints_kernel_params_cache is not None
                else None
            )
            kernel_params = gp.fit_kernel_params(
                X=normalized_params,
                Y=vals,
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                initial_kernel_params=cache,
                deterministic_objective=self._deterministic,
            )
            constraints_kernel_params.append(kernel_params)

            constraints_acqf_params.append(
                acqf.create_acqf_params(
                    acqf_type=acqf.AcquisitionFunctionType.LOG_PI,
                    kernel_params=kernel_params,
                    search_space=internal_search_space,
                    X=normalized_params,
                    Y=vals,
                    # Since 0 is the threshold value, we use the normalized value of 0.
                    max_Y=-mean / max(EPS, std),
                )
            )

        self._constraints_kernel_params_cache = constraints_kernel_params

        return constraints_acqf_params

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if study._is_multi_objective() and self._constraints_func is not None:
            raise ValueError(
                "GPSampler does not support constrained multi-objective optimization."
            )

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.values for trial in trials])
        )

        if self._kernel_params_cache_list is not None and len(
            self._kernel_params_cache_list[0].inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            # Clear cache if the search space changes.
            self._kernel_params_cache_list = None

        kernel_params_list = []
        n_objectives = standardized_score_vals.shape[-1]
        is_categorical = internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        for i in range(n_objectives):
            cache = (
                self._kernel_params_cache_list[i]
                if self._kernel_params_cache_list is not None
                else None
            )
            kernel_params_list.append(
                gp.fit_kernel_params(
                    X=normalized_params,
                    Y=standardized_score_vals[:, i],
                    is_categorical=is_categorical,
                    log_prior=self._log_prior,
                    minimum_noise=self._minimum_noise,
                    initial_kernel_params=cache,
                    deterministic_objective=self._deterministic,
                )
            )
        self._kernel_params_cache_list = kernel_params_list

        best_params: np.ndarray | None
        if self._constraints_func is None:
            if n_objectives == 1:
                assert len(kernel_params_list) == 1
                acqf_params = acqf.create_acqf_params(
                    acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
                    kernel_params=kernel_params_list[0],
                    search_space=internal_search_space,
                    X=normalized_params,
                    Y=standardized_score_vals[:, 0],
                )
                best_params = normalized_params[np.argmax(standardized_score_vals), np.newaxis]
            else:
                acqf_params_for_objectives = []
                for i in range(n_objectives):
                    acqf_params_for_objectives.append(
                        acqf.create_acqf_params(
                            acqf_type=acqf.AcquisitionFunctionType.LOG_EHVI,
                            kernel_params=kernel_params_list[i],
                            search_space=internal_search_space,
                            X=normalized_params,
                            Y=standardized_score_vals[:, i],
                        )
                    )
                acqf_params = acqf.MultiObjectiveAcquisitionFunctionParams.from_acqf_params(
                    acqf_params_for_objectives=acqf_params_for_objectives,
                    Y=standardized_score_vals,
                    n_qmc_samples=128,  # NOTE(nabenabe): The BoTorch default value.
                    qmc_seed=self._rng.rng.randint(1 << 30),
                )
                pareto_params = normalized_params[
                    _is_pareto_front(-standardized_score_vals, assume_unique_lexsorted=False)
                ]
                n_pareto_sols = len(pareto_params)
                # TODO(nabenabe): Verify the validity of this choice.
                size = min(self._n_local_search // 2, n_pareto_sols)
                chosen_indices = self._rng.rng.choice(n_pareto_sols, size=size, replace=False)
                best_params = pareto_params[chosen_indices]
        else:
            assert (
                n_objectives == len(kernel_params_list) == 1
            ), "Multi-objective has not been supported."
            constraint_vals, is_feasible = _get_constraint_vals_and_feasibility(study, trials)
            is_all_infeasible = not np.any(is_feasible)

            # TODO(kAIto47802): If is_all_infeasible, the acquisition function for the objective
            # function is ignored, so skipping the computation of kernel_params and acqf_params
            # can improve speed.
            # TODO(kAIto47802): Consider the case where all trials are feasible. We can ignore
            # constraints in this case.
            max_Y = -np.inf if is_all_infeasible else np.max(standardized_score_vals[is_feasible])
            acqf_params = acqf.create_acqf_params(
                acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
                kernel_params=kernel_params_list[0],
                search_space=internal_search_space,
                X=normalized_params,
                Y=standardized_score_vals[:, 0],
                max_Y=max_Y,
            )
            constraints_acqf_params = self._get_constraints_acqf_params(
                constraint_vals, internal_search_space, normalized_params
            )
            acqf_params = acqf.ConstrainedAcquisitionFunctionParams.from_acqf_params(
                acqf_params, constraints_acqf_params
            )
            best_params = (
                None
                if is_all_infeasible
                else normalized_params[np.argmax(standardized_score_vals[is_feasible]), np.newaxis]
            )

        normalized_param = self._optimize_acqf(acqf_params, best_params)
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

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

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)


def _get_constraint_vals_and_feasibility(
    study: Study, trials: list[FrozenTrial]
) -> tuple[np.ndarray, np.ndarray]:
    _constraint_vals = [
        study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY, ())
        for trial in trials
    ]
    if any(len(_constraint_vals[0]) != len(c) for c in _constraint_vals):
        raise ValueError("The number of constraints must be the same for all trials.")

    constraint_vals = np.array(_constraint_vals)
    assert len(constraint_vals.shape) == 2, "constraint_vals must be a 2d array."
    is_feasible = np.all(constraint_vals <= 0, axis=1)
    assert not isinstance(is_feasible, np.bool_), "MyPy Redefinition for NumPy v2.2.0."
    return constraint_vals, is_feasible
