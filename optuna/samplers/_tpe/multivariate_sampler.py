import math
import numpy as np

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.study import StudyDirection
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers._tpe.sampler import default_gamma
from optuna.samplers._tpe.sampler import default_weights
from optuna.samplers._tpe.multivariate_parzen_estimator import (
    _MultivariateParzenEstimator,
)


class MultivariateTPESampler(TPESampler):
    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: Optional[int] = None,
    ):
        super(MultivariateTPESampler, self).__init__(
            consider_prior,
            prior_weight,
            consider_magic_clip,
            consider_endpoints,
            n_startup_trials,
            n_ei_candidates,
            gamma,
            weights,
            seed,
        )
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:

        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `_MultivariateTPESampler` cannot handle distributions that contain
                # just a single value, so we skip them. Note that the parameter values
                # for such distributions are sampled in `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    distributions.UniformDistribution,
                    distributions.LogUniformDistribution,
                    distributions.DiscreteUniformDistribution,
                    distributions.IntUniformDistribution,
                    distributions.IntLogUniformDistribution,
                    distributions.CategoricalDistribution,
                ),
            ):
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:

        if search_space == {}:
            return {}

        param_names = list(search_space.keys())
        values, scores = _get_observation_pairs_multivariate(study, param_names)

        # if the number of samples is insufficient, run random trial
        n = len(scores)
        if n < self._n_startup_trials:
            ret = {}
            for param_name, param_distribution in search_space.items():
                ret[param_name] = self._random_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                )
            return ret

        # divide data into below and above
        below, above = self._split_multivariate_observation_pairs(values, scores)
        # then sample by maximizing log likelihood ratio
        mpe_below = _MultivariateParzenEstimator(
            below, search_space, self._parzen_estimator_parameters
        )
        mpe_above = _MultivariateParzenEstimator(
            above, search_space, self._parzen_estimator_parameters
        )
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihood_below = mpe_below.log_pdf(samples_below)
        log_likelihood_above = mpe_above.log_pdf(samples_below)
        ret = MultivariateTPESampler._compare_multivariate(
            samples_below, log_likelihood_below, log_likelihood_above
        )

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def _split_multivariate_observation_pairs(
        self,
        config_vals: Dict[str, List[Optional[float]]],
        loss_vals: List[Tuple[float, float]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        config_vals = {k: np.asarray(v, dtype=float) for k, v in config_vals.items()}
        loss_vals = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        # TODO(kstoneriv3): change the order of exclusion of None and splitting.
        # independent sampler in TPESampler first splits the observations and then exclude None.

        # exclude param_vals with None
        config_vals_matrix = np.array([v for v in config_vals.values()])
        index_none = np.any(np.equal(config_vals_matrix, None), axis=0)
        config_vals = {k: v[~index_none] for k, v in config_vals.items()}
        loss_vals = loss_vals[~index_none]

        n_below = self._gamma(len(loss_vals))
        index_loss_ascending = np.argsort(loss_vals)
        # np.sort is used to keep chronological order
        index_below = np.sort(index_loss_ascending[:n_below])
        index_above = np.sort(index_loss_ascending[n_below:])
        below = {}
        above = {}
        for param_name in config_vals.keys():
            below[param_name] = config_vals[param_name][index_below]
            above[param_name] = config_vals[param_name][index_above]

        return below, above

    @classmethod
    def _compare_multivariate(
        cls,
        multivariate_samples: Dict[str, np.ndarray],
        log_l: np.ndarray,
        log_g: np.ndarray,
    ) -> np.ndarray:

        sample_size = next(iter(multivariate_samples.values())).size
        if sample_size:
            score = log_l - log_g
            if sample_size != score.size:
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(
                        sample_size, score.size
                    )
                )
            best = np.argmax(score)
            return {k: v[best] for k, v in multivariate_samples.items()}
        else:
            return {k: np.asarray([]) for k in multivariate_samples.keys()}


def _get_observation_pairs_multivariate(
    study: Study, param_names: List[str]
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]:

    sign = 1
    if study.direction == StudyDirection.MAXIMIZE:
        sign = -1

    scores = []
    values = {
        param_name: [] for param_name in param_names
    }  # type: Dict[str, List[Optional[float]]]
    for trial in study._storage.get_all_trials(study._study_id, deepcopy=False):

        # extract score from trial
        if trial.state is TrialState.COMPLETE and trial.value is not None:
            score = (-float("inf"), sign * trial.value)
        elif trial.state is TrialState.PRUNED:
            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, float("inf"))
                else:
                    score = (-step, sign * intermediate_value)
            else:
                score = (float("inf"), 0.0)
        else:
            continue
        scores.append(score)

        # extract param_value from trial
        for param_name in param_names:
            param_value = None
            if param_name in trial.params:
                distribution = trial.distributions[param_name]
                param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)

    return values, scores
