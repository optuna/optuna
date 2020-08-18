import math

import numpy as np
# import scipy.special
from scipy.stats import truncnorm

from optuna import distributions
from optuna.samplers._tpe.sampler import default_gamma 
from optuna.samplers._tpe.sampler import default_weights
from optuna.samplers._tpe.multivariate_parzen_estimator import _MultivariateParzenEstimator
# from optuna.samplers import BaseSampler
# from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.samplers import IntersectionSearchSpace
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    # from typing import Any  # NOQA
    # from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA

class MultivariateTPESampler(TPESampler):
    
    def __init__(
        self,
        consider_prior=True,  # type: bool
        prior_weight=1.0,  # type: float
        consider_magic_clip=True,  # type: bool
        consider_endpoints=False,  # type: bool
        n_startup_trials=10,  # type: int
        n_ei_candidates=24,  # type: int
        gamma=default_gamma,  # type: Callable[[int], int]
        weights=default_weights,  # type: Callable[[int], np.ndarray]
        seed=None,  # type: Optional[int]
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
            seed
        )
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial",
    ) -> Dict[str, BaseDistribution]:

        search_space = {}  # type: Dict[str, BaseDistribution]
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `_MultivariateTPESampler` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
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

    ''' 
    # sample_relative from SimulatedAnnealingSampler
    # for checking input/output 

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        # copied from SA algorithm
        if search_space == {}:
            return {}

        params = {}
        for param_name, param_distribution in search_space.items():
            if not isinstance(param_distribution, distributions.UniformDistribution):
                raise NotImplementedError('Only suggest_uniform() is supported')

            neighbor_low = param_distribution.low
            neighbor_high = param_distribution.high
            # just use random sampling for now
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params
    '''

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        if search_space == {}:
            return {}

        size = (self._n_ei_candidates,)
        
        # divide data into below and above
        param_names = list(search_space.keys())
        values, scores = _get_observation_pairs_mult(study, param_names, trial)
        below, above = self._split_observation_pairs(values, scores)

        mpe_below = _MultivariateParzenEstimator(search_space, below)
        mpe_above = _MultivariateParzenEstimator(search_space, above)
        samples_below = mpe_below.sample(rng=self._rng, size=size)
        log_likelihood_below = mpe_below.log_pdf(samples_below)
        log_likelihood_above = mpe_above.log_pdf(samples_below)
        ret = MultivariateTPESampler._pick_best_sample(
            samples_below, log_likelihood_below, log_likelihood_above
        )
        return ret

    def _split_observation_pairs(
        self,
        config_vals,  # type: Dict[str, List[Optional[float]]]
        loss_vals,  # type: List[Tuple[float, float]]
    ):
        # type: (...) -> Tuple[np.ndarray, np.ndarray]

        config_vals = {k: np.asarray(v, dtype=float) for k, v in config_vals.items()}
        loss_vals = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        # TODO(kstoneriv3): change the order of exclusion of None and splitting. 
        # independent sampler in TPESampler first splits the observations and then exclude None.

        # exclude param_vals with None
        config_vals_matrix = np.concatenate([[v] for v in config_vals.values()]) 
        index_none = np.any(None == config_vals_matrix, axis=0)
        config_vals = {k: v[~index_none] for k, v in config_vals.items()}
        loss_vals = loss_vals[~index_none]

        # split 
        n_below = self._gamma(len(loss_vals))
        loss_ascending = np.argsort(loss_vals)
        index_below = np.sort(loss_ascending[:n_below])
        index_above = np.sort(loss_ascending[n_below:])
        below = {}
        above = {}
        for param_name in config_vals.keys():
            below[param_name] = config_vals[param_name][index_below]
            above[param_name] = config_vals[param_name][index_above]
        
        return below, above

    @classmethod
    def _pick_best_sample(cls, samples, log_l, log_g):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray

        if len(samples.values) == 0:
            raise ValueError("Argument samples should have at least one key.")

        sample_size = samples.values()[0].size 
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
            return {k: v[best] for k, v in samples.items()}
        else:
            return {k: np.asarray([]) for k in samples.keys()}

def _get_observation_pairs_mult(study, param_names, trial):
    # type: (Study, List[str], FrozenTrial) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]
    
    sign = 1
    if study.direction == StudyDirection.MAXIMIZE:
        sign = -1

    values = {param_name: [] for param_name in param_names}
    scores = []
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
            param_value = None  # type: Optional[float]
            if param_name in trial.params:
                distribution = trial.distributions[param_name]
                param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)

    return values, scores

