from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import ConditionalGPRegressor
from optuna._gp.gp import MultiObjectiveConditionalGPRegressor
from optuna._gp.qmc import sample_from_normal_sobol
from optuna._hypervolume import compute_hypervolume
from optuna._hypervolume import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    from typing import Protocol

    import torch

    from optuna._gp.gp import GPRegressor
    from optuna._gp.search_space import SearchSpace

    class PerSampleLogUtilityType(Protocol):
        def compute_per_sample_log_utility(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
_EPS = 1e-12  # NOTE(nabenabe): grad becomes nan when EPS=0.


def logehvi(
    Y_post: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
    non_dominated_box_lower_bounds: torch.Tensor,  # (n_boxes, n_objectives)
    non_dominated_box_intervals: torch.Tensor,  # (n_boxes, n_objectives)
) -> torch.Tensor:  # (..., )
    log_n_qmc_samples = float(np.log(Y_post.shape[-2]))
    # This function calculates Eq. (1) of https://arxiv.org/abs/2006.05078.
    # TODO(nabenabe): Adapt to Eq. (3) when we support batch optimization.
    # TODO(nabenabe): Make the calculation here more numerically stable.
    # cf. https://arxiv.org/abs/2310.20708
    # Check the implementations here:
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/safe_math.py
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/acquisition/multi_objective/logei.py#L146-L266
    diff = Y_post.unsqueeze(-2) - non_dominated_box_lower_bounds
    diff.clamp_(min=torch.tensor(_EPS, dtype=torch.float64), max=non_dominated_box_intervals)
    # NOTE(nabenabe): logsumexp with dim=-1 is for the HVI calculation and that with dim=-2 is for
    # expectation of the HVIs over the fixed_samples.
    return torch.special.logsumexp(diff.log().sum(dim=-1), dim=(-2, -1)) - log_n_qmc_samples


def _get_reference_point(Y: torch.Tensor) -> np.ndarray:
    # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
    loss_vals = -Y.numpy()
    ref_point = np.max(loss_vals, axis=0)
    return np.nextafter(np.maximum(1.1 * ref_point, 0.9 * ref_point), np.inf)


def _get_non_dominated_box_bounds(
    Y: torch.Tensor,
    ref_point: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
    loss_vals = -Y.numpy()
    # NOTE(kAIto47802): Points that do not dominate the reference point have zero
    # hypervolume contribution and must not participate in the decomposition.
    # Since Y contains Y_train, the resulting loss_vals is guaranteed to be non-empty.
    loss_vals = loss_vals[np.all(loss_vals < ref_point, axis=-1)]
    pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]
    lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
    # NOTE(nabenabe): Flip back the sign to make them compatible with maximization.
    return torch.from_numpy(-ubs), torch.from_numpy(-lbs)


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    """
    Return E_{x ~ N(0, 1)}[max(0, x+z)]
    The calculation depends on the value of z for numerical stability.
    Please refer to Eq. (9) in the following paper for more details:
        https://arxiv.org/pdf/2310.20708.pdf

    NOTE: We do not use the third condition because [-10**100, 10**100] is an overly high range.
    """
    # First condition (most z falls into this condition, so we calculate it first)
    # NOTE: ei(z) = z * cdf(z) + pdf(z)
    out = (
        (z_half := 0.5 * z) * torch.special.erfc(-_SQRT_HALF * z)  # z * cdf(z)
        + (-z_half * z).exp() * _INV_SQRT_2PI  # pdf(z)
    ).log()
    if (z_small := z[(small := z < -25)]).numel():
        # Second condition (does not happen often, so we calculate it only if necessary)
        out[small] = (
            -0.5 * z_small**2
            - _LOG_SQRT_2PI
            + (1 + _SQRT_HALF_PI * z_small * torch.special.erfcx(-_SQRT_HALF * z_small)).log()
        )
    return out


def logei(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    return standard_logei((mean - f0) / (sigma := var.sqrt_())) + sigma.log()


def _compute_mean_max_utility(log_util_vals: torch.Tensor) -> torch.Tensor:
    # Take the max over the q-batch axis, then the mean over the fixed sample axis in log space.
    # TODO(sawa3030): Consider using fatmax instead of max.
    max_log_util_vals_in_q_batch = torch.amax(log_util_vals, dim=-1)
    return torch.special.logsumexp(max_log_util_vals_in_q_batch, dim=-1) - math.log(
        max_log_util_vals_in_q_batch.shape[-1]
    )


class _PerSampleParetoBaseline:
    """Builds per-QMC Pareto baselines from ``Y_baseline`` and the provided fantasy observations.

    ``fantasy_samples_per_qmc[i]`` is the fantasy observation matrix for the i-th QMC sample,
    with shape ``(n_running_for_sample, n_objectives)``. These fantasy observations may already
    be filtered before construction, for example to include only fantasies sampled as feasible
    under constraints. For each QMC sample, this class combines ``Y_baseline`` with the
    corresponding fantasy observations, computes the non-dominated box decomposition, and
    evaluates per-sample log HVI terms from that decomposition.
    """

    def __init__(
        self, Y_baseline: torch.Tensor, fantasy_samples_per_qmc: list[torch.Tensor]
    ) -> None:
        self.ref_point = _get_reference_point(Y_baseline)
        self._Y_baseline = Y_baseline
        self._Y_per_qmc = [
            torch.cat([Y_baseline, fantasy], dim=0) if fantasy.shape[0] > 0 else Y_baseline
            for fantasy in fantasy_samples_per_qmc
        ]
        bounds = [_get_non_dominated_box_bounds(Y, self.ref_point) for Y in self._Y_per_qmc]
        self._non_dominated_box_lower_bounds = torch.nn.utils.rnn.pad_sequence(
            [lbs for lbs, _ in bounds], batch_first=True
        )
        self._non_dominated_box_intervals = torch.nn.utils.rnn.pad_sequence(
            [(ubs - lbs).clamp_min_(_EPS) for lbs, ubs in bounds],
            batch_first=True,
            padding_value=_EPS,
        )
        self._per_sample_log_fantasy_hvi = None

    def compute_per_sample_log_hvi(self, Y_post: torch.Tensor) -> torch.Tensor:
        diff = Y_post.unsqueeze(-2) - self._non_dominated_box_lower_bounds
        # NOTE(nabenabe): EPS prevents gradients from becoming nan.
        diff.clamp_(
            min=torch.tensor(_EPS, dtype=torch.float64),
            max=self._non_dominated_box_intervals,
        )
        return torch.special.logsumexp(diff.log().sum(dim=-1), dim=-1)

    def compute_per_sample_log_fantasy_hvi(self) -> None:
        """Return ``log HVI(fantasy | Y_baseline)`` per QMC sample, constant w.r.t. the candidate.

        Computed on demand rather than in ``__init__`` because only the constrained acqf needs
        it and it costs one exact hypervolume computation per QMC sample.
        """
        baseline_hv = compute_hypervolume(
            -self._Y_baseline.numpy(),
            self.ref_point,
            assume_pareto=False,
        )
        fantasy_loss_vals = [
            (-Y.numpy())[np.all(-Y.numpy() < self.ref_point, axis=-1)] for Y in self._Y_per_qmc
        ]
        hvis = (
            np.array(
                [
                    compute_hypervolume(
                        loss_vals,
                        self.ref_point,
                        assume_pareto=False,
                    )
                    for loss_vals in fantasy_loss_vals
                ]
            )
            - baseline_hv
        )
        log_hvis = np.full_like(hvis, -np.inf)
        np.log(hvis, out=log_hvis, where=hvis > 0.0)
        self._per_sample_log_fantasy_hvi = torch.from_numpy(log_hvis)

    def get_per_sample_log_fantasy_hvi(self) -> torch.Tensor:
        assert self._per_sample_log_fantasy_hvi is not None, (
            "Call compute_per_sample_log_fantasy_hvi() first."
        )
        return self._per_sample_log_fantasy_hvi


class BaseAcquisitionFunc(ABC):
    def __init__(self, length_scales: np.ndarray, search_space: SearchSpace) -> None:
        self.length_scales = length_scales
        self.search_space = search_space

    @abstractmethod
    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.eval_acqf(torch.from_numpy(x)).detach().numpy()

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x).requires_grad_(True)
        val = self.eval_acqf(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore


class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        # If there are no feasible trials, max_Y is set to -np.inf.
        # If max_Y is set to -np.inf, we set logEI to zero to ignore it.
        return (
            logei(mean=mean, var=var + self._stabilizing_noise, f0=self._threshold)
            if not np.isneginf(self._threshold)
            else torch.zeros(x.shape[:-1], dtype=torch.float64)
        )


class qLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        n_qmc_samples: int,
        qmc_seed: int,
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._threshold = threshold
        self._cond_gpr = ConditionalGPRegressor(
            gpr=gpr,
            X_running=torch.from_numpy(normalized_params_of_running_trials),
            n_qmc_samples=n_qmc_samples,
            qmc_seed=qmc_seed,
            stabilizing_noise=stabilizing_noise,
        )
        n_running = len(normalized_params_of_running_trials)
        self._per_sample_log_util_shape = (n_qmc_samples, n_running + 1)
        super().__init__(gpr.length_scales, search_space)

    def compute_per_sample_log_utility(self, x: torch.Tensor) -> torch.Tensor:
        if np.isneginf(self._threshold):
            return_tensor_shape = x.shape[:-1] + self._per_sample_log_util_shape
            return torch.zeros(return_tensor_shape, dtype=torch.float64)
        # NOTE(nabenabe): See Eq. (10) of https://arxiv.org/pdf/2310.20708
        y_post = self._cond_gpr.sample_joint_posterior(x)
        return (y_post - self._threshold).clamp_min_(_EPS).log()

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return _compute_mean_max_utility(self.compute_per_sample_log_utility(x))


class LogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold

        if normalized_params_of_running_trials is not None:
            normalized_params_of_running_trials_tensor = torch.from_numpy(
                normalized_params_of_running_trials
            )

            # NOTE(sawa3030): To handle running trials, the Kriging Believer strategy is
            # currently implemented, as it is simple and performs well in our benchmarks.
            # We plan to implement Monte-Carlo based approaches (e.g., BoTorch’s fantasize)
            # in the near future.
            # See https://github.com/optuna/optuna/pull/6481 for details.
            # For background on the Constant Liar and Kriging Believer strategies, see
            # Ginsbourger et al., "Kriging Is Well-Suited to Parallelize Optimization" (2010).

            self._gpr.append_running_data(
                normalized_params_of_running_trials_tensor,
                gpr.posterior(normalized_params_of_running_trials_tensor)[0],
            )
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        # Return the integral of N(mean, var) from f0 to inf.
        # This is identical to the integral of N(0, 1) from (f0-mean)/sigma to inf.
        # Return E_{y ~ N(mean, var)}[bool(y >= f0)]
        mean, var = self._gpr.posterior(x)
        sigma = torch.sqrt(var + self._stabilizing_noise)
        # NOTE(nabenabe): integral from a to b of f(x) is integral from -b to -a of f(-x).
        return torch.special.log_ndtr((mean - self._threshold) / sigma)


class qLogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        n_qmc_samples: int,
        qmc_seed: int,
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
        tau: float = 1e-2,
    ) -> None:
        self._threshold = threshold
        self._tau = tau
        self._cond_gpr = ConditionalGPRegressor(
            gpr=gpr,
            X_running=torch.from_numpy(normalized_params_of_running_trials),
            n_qmc_samples=n_qmc_samples,
            qmc_seed=qmc_seed,
            stabilizing_noise=stabilizing_noise,
        )
        super().__init__(gpr.length_scales, search_space)

    def compute_per_sample_log_utility(self, x: torch.Tensor) -> torch.Tensor:
        y_post = self._cond_gpr.sample_joint_posterior(x)
        return torch.nn.functional.logsigmoid((y_post - self._threshold) / self._tau)

    def compute_candidate_log_utility(self, x: torch.Tensor) -> torch.Tensor:
        y_post = self._cond_gpr.sample_joint_posterior(x, return_fantasy=False)
        return torch.nn.functional.logsigmoid((y_post - self._threshold) / self._tau)

    def is_fantasy_above_threshold(self) -> torch.Tensor:
        return self._cond_gpr.get_fantasy_samples() >= self._threshold

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return _compute_mean_max_utility(self.compute_per_sample_log_utility(x))


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean + torch.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean - torch.sqrt(self._beta * var)


class LogCEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = LogEI(gpr, search_space, threshold, stabilizing_noise)
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                None,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(kAIto47802): Handle the infeasible case inside `LogCEI` instead of `LogEI`.
        return self._acqf.eval_acqf(x) + sum(
            acqf.eval_acqf(x) for acqf in self._constraints_acqf_list
        )


class qLogCEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        n_qmc_samples: int,
        qmc_seed: int,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf: PerSampleLogUtilityType = qLogEI(
            gpr,
            search_space,
            threshold,
            n_qmc_samples,
            qmc_seed,
            normalized_params_of_running_trials,
            stabilizing_noise,
        )
        self._constraints_acqf_list: list[PerSampleLogUtilityType] = [
            qLogPI(
                gpr=constraint_gpr,
                search_space=search_space,
                threshold=constraint_threshold,
                n_qmc_samples=n_qmc_samples,
                qmc_seed=qmc_seed + i + 1,
                normalized_params_of_running_trials=normalized_params_of_running_trials,
                stabilizing_noise=stabilizing_noise,
            )
            for i, (constraint_gpr, constraint_threshold) in enumerate(
                zip(
                    constraints_gpr_list,
                    constraints_threshold_list,
                )
            )
        ]
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        log_feasible_improvement = self._acqf.compute_per_sample_log_utility(x) + sum(
            acqf.compute_per_sample_log_utility(x) for acqf in self._constraints_acqf_list
        )
        return _compute_mean_max_utility(log_feasible_improvement)


class LogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_train: torch.Tensor,
        n_qmc_samples: int,
        qmc_seed: int,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._stabilizing_noise = stabilizing_noise
        self._gpr_list = gpr_list
        if normalized_params_of_running_trials is not None:
            normalized_params_of_running_trials_tensor = torch.from_numpy(
                normalized_params_of_running_trials
            )

            # NOTE(sawa3030): To handle running trials, the Kriging Believer strategy is
            # currently implemented, as it is simple and performs well in our benchmarks.
            # We plan to implement Monte-Carlo based approaches (e.g., BoTorch’s fantasize)
            # in the near future.
            # See https://github.com/optuna/optuna/pull/6481 for details.
            # For background on the Constant Liar and Kriging Believer strategies, see
            # Ginsbourger et al., "Kriging Is Well-Suited to Parallelize Optimization" (2010).

            for gpr in self._gpr_list:
                gpr.append_running_data(
                    normalized_params_of_running_trials_tensor,
                    gpr.posterior(normalized_params_of_running_trials_tensor)[0],
                )

        self._fixed_samples = sample_from_normal_sobol(
            dim=Y_train.shape[-1], n_samples=n_qmc_samples, seed=qmc_seed
        )
        ref_point = _get_reference_point(Y_train)
        self._non_dominated_box_lower_bounds, non_dominated_box_upper_bounds = (
            _get_non_dominated_box_bounds(Y_train, ref_point)
        )
        self._non_dominated_box_intervals = (
            non_dominated_box_upper_bounds - self._non_dominated_box_lower_bounds
        ).clamp_min_(_EPS)
        # Since all the objectives are equally important, we simply use the mean of
        # inverse of squared mean lengthscales over all the objectives.
        # inverse_squared_lengthscales is used in optim_mixed.py.
        # cf. https://github.com/optuna/optuna/blob/v4.3.0/optuna/_gp/optim_mixed.py#L200-L209
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        Y_post = []
        for i, gpr in enumerate(self._gpr_list):
            mean, var = gpr.posterior(x)
            stdev = torch.sqrt(var + self._stabilizing_noise)
            # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes
            # deterministic, making it possible to optimize the acqf by l-BFGS.
            # Sobol is better than the standard Monte-Carlo w.r.t. the approximation stability.
            # cf. Appendix D of https://arxiv.org/pdf/2006.05078
            Y_post.append(mean[..., None] + stdev[..., None] * self._fixed_samples[..., i])

        # NOTE(nabenabe): Use the following once multi-task GP is supported.
        # L = torch.linalg.cholesky(cov)
        # Y_post = means[..., None, :] + torch.einsum("...MM,SM->...SM", L, fixed_samples)
        return logehvi(
            Y_post=torch.stack(Y_post, dim=-1),
            non_dominated_box_lower_bounds=self._non_dominated_box_lower_bounds,
            non_dominated_box_intervals=self._non_dominated_box_intervals,
        )


class qLogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_train: torch.Tensor,
        n_qmc_samples: int,
        qmc_seed: int,
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._cond_gpr = MultiObjectiveConditionalGPRegressor(
            gpr_list=gpr_list,
            X_running=torch.from_numpy(normalized_params_of_running_trials),
            n_qmc_samples=n_qmc_samples,
            qmc_seed=qmc_seed,
            stabilizing_noise=stabilizing_noise,
        )
        self._baseline = _PerSampleParetoBaseline(Y_train, list(self._cond_gpr.fantasy_samples))
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def compute_per_sample_log_utility(self, x: torch.Tensor) -> torch.Tensor:
        return self._baseline.compute_per_sample_log_hvi(
            self._cond_gpr.sample_candidate_posterior(x)
        )

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        log_util_vals = self.compute_per_sample_log_utility(x)
        return torch.special.logsumexp(log_util_vals, dim=-1) - math.log(log_util_vals.shape[-1])


class LogCEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_feasible: torch.Tensor | None,
        n_qmc_samples: int,
        qmc_seed: int,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = (
            LogEHVI(
                gpr_list,
                search_space,
                Y_feasible,
                n_qmc_samples,
                qmc_seed,
                None,
                stabilizing_noise,
            )
            if Y_feasible is not None
            else None
        )
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                None,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        # Since all the objectives are equally important, we simply use the mean of
        # inverse of squared mean lengthscales over all the objectives.
        # inverse_squared_lengthscales is used in optim_mixed.py.
        # cf. https://github.com/optuna/optuna/blob/v4.3.0/optuna/_gp/optim_mixed.py#L200-L209
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        constraints_acqf_values = sum(acqf.eval_acqf(x) for acqf in self._constraints_acqf_list)
        if self._acqf is None:
            return cast("torch.Tensor", constraints_acqf_values)
        return constraints_acqf_values + self._acqf.eval_acqf(x)


class qLogCEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_feasible: torch.Tensor | None,
        n_qmc_samples: int,
        qmc_seed: int,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray,
        stabilizing_noise: float = 1e-12,
        tau: float = 1e-2,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._constraints_acqf_list: list[qLogPI] = [
            qLogPI(
                gpr=constraint_gpr,
                search_space=search_space,
                threshold=constraint_threshold,
                n_qmc_samples=n_qmc_samples,
                qmc_seed=qmc_seed + len(gpr_list) + i,
                normalized_params_of_running_trials=normalized_params_of_running_trials,
                stabilizing_noise=stabilizing_noise,
                tau=tau,
            )
            for i, (constraint_gpr, constraint_threshold) in enumerate(
                zip(constraints_gpr_list, constraints_threshold_list)
            )
        ]

        # NOTE(sawa3030): qLogCEHVI evaluates HVI(Y_running U Y_candidate | Y_train)
        # in log space using HVI(Y_candidate | Y_running U Y_train) + HVI(Y_running | Y_train).
        # qLogEHVI only needs the first term because HVI(Y_running | Y_train) is constant with
        # respect to x, whereas qLogCEHVI keeps both terms. Here, self._acqf computes
        # the first term and self._baseline.get_per_sample_log_fantasy_hvi() gets the second
        # term per fantasy sample.
        self._cond_gpr: MultiObjectiveConditionalGPRegressor | None = None
        self._baseline: _PerSampleParetoBaseline | None = None
        if Y_feasible is not None:
            is_fantasy_feasible = torch.all(
                torch.stack(
                    [acqf.is_fantasy_above_threshold() for acqf in self._constraints_acqf_list],
                    dim=-1,
                ),
                dim=-1,
            )
            self._cond_gpr = MultiObjectiveConditionalGPRegressor(
                gpr_list=gpr_list,
                X_running=torch.from_numpy(normalized_params_of_running_trials),
                n_qmc_samples=n_qmc_samples,
                qmc_seed=qmc_seed,
                stabilizing_noise=stabilizing_noise,
            )
            self._baseline = _PerSampleParetoBaseline(
                Y_feasible,
                [
                    fantasy[is_feasible]
                    for fantasy, is_feasible in zip(
                        self._cond_gpr.fantasy_samples, is_fantasy_feasible
                    )
                ],
            )
            self._baseline.compute_per_sample_log_fantasy_hvi()
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        constraint_log_feasibility = sum(
            acqf.compute_candidate_log_utility(x) for acqf in self._constraints_acqf_list
        )
        if self._baseline is None or self._cond_gpr is None:
            return torch.special.logsumexp(constraint_log_feasibility, dim=-1) - math.log(
                constraint_log_feasibility.shape[-1]
            )
        log_feasible_improvement = torch.logaddexp(
            self._baseline.get_per_sample_log_fantasy_hvi(),
            constraint_log_feasibility
            + self._baseline.compute_per_sample_log_hvi(
                self._cond_gpr.sample_candidate_posterior(x)
            ),
        )
        return torch.special.logsumexp(log_feasible_improvement, dim=-1) - math.log(
            log_feasible_improvement.shape[-1]
        )
