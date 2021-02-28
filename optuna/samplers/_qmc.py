# Especially, we should re-consider how i4_sobol handles its seed using global variable.

from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional

import scipy

import optuna
from optuna import distributions
from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)

_SUGGESTED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)

_NUMERICAL_DISTRIBUTIONS = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.DiscreteUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
)


# It is recommended that we take the number of samples as power of two
# to properly explit sobol sequence.
# However, sobol sampler works iteratively anyways, so can use n_trials
# which is not power of two.


class QMCSampler(BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        qmc_type="sobol",
        independent_sampler=None,
        *,
        search_space: Optional[Dict[str, BaseDistribution]] = None,
        warn_independent_sampling: bool = True,
    ) -> None:

        # handle dimentions
        # handle sample size that is not power of 2
        # probably we do not have to
        self._seed = seed
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._qmc_type = qmc_type
        self._qmc_engine = None
        # TODO(kstoneriv3): make sure that search_space is either None or valid search space.
        self._initial_search_space = search_space
        self._warn_independent_sampling = warn_independent_sampling

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        if self._initial_search_space is not None:
            return self._initial_search_space

        past_trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
        past_trials = [t for t in past_trials if t.state in _SUGGESTED_STATES]
        past_trials = sorted(past_trials, key=lambda t: t._trial_id)

        # The first trial is sampled by the independent sampler.
        if len(past_trials) == 0:
            return {}
        # If an initial trial was already made,
        # construct search_space of this sampler from the initial trial.
        else:
            self._initial_search_space = self._infer_initial_search_space(past_trials[0])
            return self._initial_search_space

    def _infer_initial_search_space(self, trial: FrozenTrial) -> Dict[str, BaseDistribution]:

        search_space = OrderedDict()  # type: OrderedDict[str, BaseDistribution]
        for param_name, distribution in trial.distributions.items():
            if not isinstance(distribution, _NUMERICAL_DISTRIBUTIONS):
                if self._warn_independent_sampling:
                    self._log_independent_sampling(trial, param_name)
                continue
            search_space[param_name] = distribution

        return search_space

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `QMCSampler` "
            "(optimization performance may be degraded). "
            "`QMCSampler` does not support dynamic search space or `CategoricalDistribution`. "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `QMCSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _reset_qmc_engine(self, d: int):

        # Lazy import because the `scipy.stats.qmc` is slow to import.
        import scipy.stats.qmc

        self._samples_count = 0  # The number of samples, taken from the engine.
        scramble = False if self._seed is None else True
        if self._qmc_type == "sobol":
            self._qmc_engine = scipy.stats.qmc.Sobol(d, seed=self._seed, scramble=scramble)
        elif self._qmc_type == "halton":
            self._qmc_engine = scipy.stats.qmc.Halton(d, seed=self._seed, scramble=scramble)
        elif self._qmc_type == "LHS":  # Latin Hypercube Sampling
            self._qmc_engine = scipy.stats.qmc.Latin(d, seed=self._seed)
        elif self._qmc_type == "OA-LHS":  # Orthogonal array-based Latin hypercube sampling
            self._qmc_engine = scipy.stats.qmc.OrthogonalLatinHypercube(d, seed=self._seed)
        else:
            message = (
                "The `qmc_type`, {}, is not a valid. "
                "It must be one of sobol, halton, LHS, and OA-LHS."
            )
            raise ValueError(message)

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

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        if search_space == {}:
            return {}

        assert self._initial_search_space is not None

        if self._qmc_engine is None:
            n_initial_params = len(self._initial_search_space)
            self._reset_qmc_engine(n_initial_params)

        qmc_id = self._find_qmc_id(study, trial)
        forward_size = qmc_id - self._qmc_engine.num_generated  # `qmc_id` starts from 0.
        self._qmc_engine.fast_forward(forward_size)
        sample = self._qmc_engine.random(1)

        trans = _SearchSpaceTransform(search_space)
        sample = scipy.stats.qmc.scale(sample, trans.bounds[:, 0], trans.bounds[:, 1])
        sample = trans.untransform(sample[0, :])

        return sample

    def _find_qmc_id(self, study: Study, trial: FrozenTrial):
        # TODO(kstoneriv3): Following try-except block assumes that the block is
        # an atomic transaction. # This ensures that each qmc_id is sampled at least once.
        key_qmc_id = "{}_last_qmc_id".format(self._qmc_type)
        try:
            qmc_id = study._storage.get_study_system_attrs(study._study_id)[key_qmc_id]
            qmc_id += 1
            study._storage.set_study_system_attr(study._study_id, key_qmc_id, qmc_id)
        except KeyError:
            study._storage.set_study_system_attr(study._study_id, key_qmc_id, 0)
            qmc_id = 0

        return qmc_id
