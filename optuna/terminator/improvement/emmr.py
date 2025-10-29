from __future__ import annotations

import math
import sys
from typing import cast
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import intersection_search_space
from optuna.study import StudyDirection
from optuna.terminator.improvement.evaluator import _compute_standardized_regret_bound
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import scipy.stats as scipy_stats
    import torch

    from optuna._gp import acqf as acqf_module
    from optuna._gp import gp
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp = _LazyImport("optuna._gp.gp")
    acqf_module = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    scipy_stats = _LazyImport("scipy.stats")

MARGIN_FOR_NUMARICAL_STABILITY = 0.1


@experimental_class("4.0.0")
class EMMREvaluator(BaseImprovementEvaluator):
    """Evaluates a kind of regrets, called the Expected Minimum Model Regret(EMMR).

    EMMR is an upper bound of "expected minimum simple regret" in the optimization process.

    Expected minimum simple regret is a quantity that converges to zero only if the
    optimization process has found the global optima.

    For further information about expected minimum simple regret and the algorithm,
    please refer to the following paper:

    - `A stopping criterion for Bayesian optimization by the gap of expected minimum simple
      regrets <https://proceedings.mlr.press/v206/ishibashi23a.html>`__

    Also, there is our blog post explaining this evaluator:

    - `Introducing A New Terminator: Early Termination of Black-box Optimization Based on
      Expected Minimum Model Regret
      <https://medium.com/optuna/introducing-a-new-terminator-early-termination-of-black-box-optimization-based-on-expected-9a660774fcdb>`__

    Args:
        deterministic_objective:
            A boolean value which indicates whether the objective function is deterministic.
            Default is :obj:`False`.
        delta:
            A float number related to the criterion for termination. Default to 0.1.
            For further information about this parameter, please see the aforementioned paper.
        min_n_trials:
            A minimum number of complete trials to compute the criterion. Default to 2.
        seed:
            A random seed for EMMREvaluator.

    Example:

        .. testcode::

            import optuna
            from optuna.terminator import EMMREvaluator
            from optuna.terminator import MedianErrorEvaluator
            from optuna.terminator import Terminator

            sampler = optuna.samplers.TPESampler(seed=0)
            study = optuna.create_study(sampler=sampler, direction="minimize")
            emmr_improvement_evaluator = EMMREvaluator()
            median_error_evaluator = MedianErrorEvaluator(emmr_improvement_evaluator)
            terminator = Terminator(
                improvement_evaluator=emmr_improvement_evaluator,
                error_evaluator=median_error_evaluator,
            )


            for i in range(1000):
                trial = study.ask()

                ys = [trial.suggest_float(f"x{i}", -10.0, 10.0) for i in range(5)]
                value = sum(ys[i] ** 2 for i in range(5))

                study.tell(trial, value)

                if terminator.should_terminate(study):
                    # Terminated by Optuna Terminator!
                    break

    """

    def __init__(
        self,
        deterministic_objective: bool = False,
        delta: float = 0.1,
        min_n_trials: int = 2,
        seed: int | None = None,
    ) -> None:
        if min_n_trials <= 1 or not np.isfinite(min_n_trials):
            raise ValueError("`min_n_trials` is expected to be a finite integer more than one.")

        self._deterministic = deterministic_objective
        self._delta = delta
        self.min_n_trials = min_n_trials
        self._rng = LazyRandomState(seed)

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:

        optuna_search_space = intersection_search_space(trials)
        complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]

        if len(complete_trials) < self.min_n_trials:
            return sys.float_info.max * MARGIN_FOR_NUMARICAL_STABILITY  # Do not terminate.

        search_space = gp_search_space.SearchSpace(optuna_search_space)
        normalized_params = search_space.get_normalized_params(complete_trials)
        if not search_space.dim:
            warnings.warn(
                f"{self.__class__.__name__} cannot consider any search space."
                "Termination will never occur in this study."
            )
            return sys.float_info.max * MARGIN_FOR_NUMARICAL_STABILITY  # Do not terminate.

        len_trials = len(complete_trials)
        assert normalized_params.shape == (len_trials, search_space.dim)

        # _gp module assumes that optimization direction is maximization
        sign = -1 if study_direction == StudyDirection.MINIMIZE else 1
        score_vals = np.array([cast(float, t.value) for t in complete_trials]) * sign
        score_vals = gp.warn_and_convert_inf(score_vals)
        standarized_score_vals = (score_vals - score_vals.mean()) / max(
            sys.float_info.min, score_vals.std()
        )

        assert len(standarized_score_vals) == len(normalized_params)

        gpr_t1 = gp.fit_kernel_params(  # Fit kernel with up to (t-1)-th observation
            X=normalized_params[..., :-1, :],
            Y=standarized_score_vals[:-1],
            is_categorical=search_space.is_categorical,
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            gpr_cache=None,
            deterministic_objective=self._deterministic,
        )

        gpr_t = gp.fit_kernel_params(  # Fit kernel with up to t-th observation
            X=normalized_params,
            Y=standarized_score_vals,
            is_categorical=search_space.is_categorical,
            log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
            gpr_cache=gpr_t1,
            deterministic_objective=self._deterministic,
        )

        theta_t_star_index = int(np.argmax(standarized_score_vals))
        theta_t1_star_index = int(np.argmax(standarized_score_vals[:-1]))
        theta_t_star = normalized_params[theta_t_star_index, :]
        theta_t1_star = normalized_params[theta_t1_star_index, :]
        cov_t_between_theta_t_star_and_theta_t1_star = _compute_gp_posterior_cov_two_thetas(
            normalized_params, gpr_t, theta_t_star_index, theta_t1_star_index
        )
        # Use gpr_t instead of gpr_t1 because KL Div. requires the same prior for both posterior.
        # cf. Sec. 4.4 of https://proceedings.mlr.press/v206/ishibashi23a/ishibashi23a.pdf
        mu_t1_theta_t_with_nu_t, variance_t1_theta_t_with_nu_t = _compute_gp_posterior(
            normalized_params[-1, :], gpr_t
        )
        _, variance_t_theta_t1_star = _compute_gp_posterior(theta_t1_star, gpr_t)
        mu_t_theta_t_star, variance_t_theta_t_star = _compute_gp_posterior(theta_t_star, gpr_t)
        mu_t1_theta_t1_star, _ = _compute_gp_posterior(theta_t1_star, gpr_t1)

        y_t = standarized_score_vals[-1]
        kappa_t1 = _compute_standardized_regret_bound(
            gpr_t1,
            search_space,
            normalized_params[:-1, :],
            standarized_score_vals[:-1],
            self._delta,
            rng=self._rng.rng,
        )

        theorem1_delta_mu_t_star = mu_t1_theta_t1_star - mu_t_theta_t_star

        alg1_delta_r_tilde_t_term1 = theorem1_delta_mu_t_star

        theorem1_v = math.sqrt(
            max(
                1e-10,
                variance_t_theta_t_star
                - 2.0 * cov_t_between_theta_t_star_and_theta_t1_star
                + variance_t_theta_t1_star,
            )
        )
        theorem1_g = (mu_t_theta_t_star - mu_t1_theta_t1_star) / theorem1_v

        alg1_delta_r_tilde_t_term2 = theorem1_v * scipy_stats.norm.pdf(theorem1_g)
        alg1_delta_r_tilde_t_term3 = theorem1_v * theorem1_g * scipy_stats.norm.cdf(theorem1_g)

        _lambda = prior.DEFAULT_MINIMUM_NOISE_VAR**-1
        eq4_rhs_term1 = 0.5 * math.log(1.0 + _lambda * variance_t1_theta_t_with_nu_t)
        eq4_rhs_term2 = (
            -0.5 * variance_t1_theta_t_with_nu_t / (variance_t1_theta_t_with_nu_t + _lambda**-1)
        )
        eq4_rhs_term3 = (
            0.5
            * variance_t1_theta_t_with_nu_t
            * (y_t - mu_t1_theta_t_with_nu_t) ** 2
            / (variance_t1_theta_t_with_nu_t + _lambda**-1) ** 2
        )

        alg1_delta_r_tilde_t_term4 = kappa_t1 * math.sqrt(
            0.5 * (eq4_rhs_term1 + eq4_rhs_term2 + eq4_rhs_term3)
        )

        return min(
            sys.float_info.max * 0.5,
            alg1_delta_r_tilde_t_term1
            + alg1_delta_r_tilde_t_term2
            + alg1_delta_r_tilde_t_term3
            + alg1_delta_r_tilde_t_term4,
        )


def _compute_gp_posterior(x_params: np.ndarray, gpr: gp.GPRegressor) -> tuple[float, float]:
    # best_params or normalized_params[..., -1, :]
    mean, var = gpr.posterior(torch.from_numpy(x_params))
    return mean.item(), var.item()


def _compute_gp_posterior_cov_two_thetas(
    normalized_params: np.ndarray, gpr: gp.GPRegressor, theta1_index: int, theta2_index: int
) -> float:  # cov

    if theta1_index == theta2_index:
        return _compute_gp_posterior(normalized_params[theta1_index], gpr)[1]

    _, covar = gpr.posterior(
        torch.from_numpy(normalized_params[[theta1_index, theta2_index]]), joint=True
    )
    assert covar.shape == (2, 2)
    return covar[0, 1].item()
