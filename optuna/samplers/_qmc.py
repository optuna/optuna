from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional

import numpy
import scipy

import optuna
from optuna import distributions
from optuna import logging
from optuna._experimental import experimental
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


@experimental("2.x.0")  # TODO(kstoneriv3)
class QMCSampler(BaseSampler):
    """A Quasi Monte Carlo Sampler that generates low-discrepancy sequences.

    Quasi Monte Carlo (QMC) sequences are designed to have low-discrepancies than
    standard random seqeunces. They are known to perform better than the standard
    randam sequences in hyperparameter optimization.

    For further information about the use of QMC sequences for hyperparameter optimization,
    please refer to the following paper:
    - Bergstra, James, and Yoshua Bengio. Random search for hyper-parameter optimization.
      Journal of machine learning research 13.2, 2012.
      <https://jmlr.org/papers/v13/bergstra12a.html>`_

    We use the QMC implementations in Scipy. For the details of the QMC algorithm,
    see the Scipy API references on `scipy.stats.qmc
    <https://scipy.github.io/devdocs/stats.qmc.html>`_.

    .. note:
        Please note that this sampler does not support CategoricalDistribution.
        If your search space contains categorical parameters, it samples the catagorical
        parameters by its `independent_sampler` without using QMC algorithm.

    Args:
        qmc_type:
            The type of QMC sequence to be sampled. This must be one of
            `"sobol"`, `"halton"`, `"LHS"` and `"OA-LHS"`. Default is `"sobol"`.

        scramble:
            In cases ``qmc_type`` is `"sobol"` or `"halton"`, if this option is :obj:`True`,
            scrambling (randomization) is applied to the QMC sequences.

        seed:
            A seed for `QMCSampler`. When the ``qmc_type`` is `"sobol"` or `"halton"`,
            this argument is used only when `scramble` is :obj:`True`. If this is :obj:`None`,
            the seed is initialized randomly. Default is :obj:`None`.

            .. note::
                When using multiple :class:`~optuna.samplers.QMCSampler`'s in parallel and/or
                distributed optimization, all the samplers must share the same seed when the
                `scrambling` is enabled. Otherwise, the low-discrepancy property of the samples
                will be degraded.

        search_space:
            The search space of the sampler.

            If this argument is not provided and there are prior
            trials in the study, :class:`~optuna.samplers.QMCSamper` infers its search space using
            the first trial of the study.

            If this argument if not provided and the study has no
            prior trials, :class:`~optuna.samplers.QMCSampler` samples the first trial using its
            `_independent_sampler` and then infers the search space in the second trial.

            .. note::
                As mentioned above, the search space of the :class:`~optuna.sampler.QMCSampler` is
                determined by argument ``search_space`` or the first trial of the study. Once
                the search space is determined, it cannot be changed afterwards.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The first trial of the study and the parameters not contained in the
            relative search space are sampled by this sampler.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`~optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are sampled via an
            independent sampler in most cases, so no warning messages are emitted in such cases.

    Raises:
        ValueError:
            If ``qmc_type`` is not one of 'sobol', 'halton', 'LHS' or 'OA-LHS'.

    .. note::
        Added in v2.x.0 TODO(kstoneriv3)as an experimental feature. The interface may change in
        newer versions without prior notice.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.QMCSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x ** 2 + y


            sampler = optuna.samplers.QMCSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    """

    def __init__(
        self,
        *,
        qmc_type: str = "sobol",
        scramble: bool = False,
        seed: Optional[int] = None,
        search_space: Optional[Dict[str, BaseDistribution]] = None,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        warn_asyncronous_seeding: bool = True,
    ) -> None:

        self._scramble = scramble
        self._seed = seed or numpy.random.MT19937().random_raw()
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._qmc_type = qmc_type
        self._cached_qmc_engine = None
        # TODO(kstoneriv3): make sure that search_space is either None or valid search space.
        # also make sure that it is OrderedDict
        self._initial_search_space = search_space
        self._warn_independent_sampling = warn_independent_sampling

        if (seed is None) and warn_asyncronous_seeding:
            # Sobol/Halton sequences without scrambling do not use seed.
            if not (qmc_type in ("sobol", "halton") and (scramble is False)):
                self._log_asyncronous_seeding()

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()
        self._seed = numpy.random.MT19937().random_raw()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        if self._initial_search_space is not None:
            return self._initial_search_space

        past_trials = study._storage.get_all_trials(
            study._study_id, states=_SUGGESTED_STATES, deepcopy=False
        )
        past_trials = sorted(past_trials, key=lambda t: t._trial_id)

        # The initial trial is sampled by the independent sampler.
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

    @staticmethod
    def _log_asyncronous_seeding() -> None:
        _logger.warning(
            "No seed is provided for `QMCSampler` and the seed is set randomly. "
            "If you are running multiple `QMCSampler`s in parallel and/or distributed "
            " environment, the same seed must be used in all samplers to ensure that resulting "
            "samples are taken from the same QMC sequence. "
            "You can suppress this warning by setting `warn_asyncronous_seeding` "
            "to `False` in the constructor of `QMCSampler`, "
            "if this random seeding is intended behavior."
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        if self._initial_search_space is not None:
            if self._warn_independent_sampling:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        if search_space == {}:
            return {}

        sample = self._sample_qmc(study, trial, search_space)
        trans = _SearchSpaceTransform(search_space)
        sample = scipy.stats.qmc.scale(sample, trans.bounds[:, 0], trans.bounds[:, 1])
        sample = trans.untransform(sample[0, :])

        return sample

    def _sample_qmc(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> numpy.ndarray:

        # Lazy import because the `scipy.stats.qmc` is slow to import.
        import scipy.stats.qmc

        sample_id = self._find_sample_id(study, trial, search_space)
        d = len(search_space)

        # Use cached `qmc_engine` or construct a new one.
        if self._is_engine_cached(d, sample_id):
            qmc_engine = self._cached_qmc_engine
        else:
            if self._qmc_type == "sobol":
                qmc_engine = scipy.stats.qmc.Sobol(d, seed=self._seed, scramble=self._scramble)
            elif self._qmc_type == "halton":
                qmc_engine = scipy.stats.qmc.Halton(d, seed=self._seed, scramble=self._scramble)
            elif self._qmc_type == "LHS":  # Latin Hypercube Sampling
                qmc_engine = scipy.stats.qmc.LatinHypercube(d, seed=self._seed)
            elif self._qmc_type == "OA-LHS":  # Orthogonal array-based Latin hypercube sampling
                qmc_engine = scipy.stats.qmc.OrthogonalLatinHypercube(d, seed=self._seed)
            else:
                message = (
                    f"The `qmc_type`, {self._qmc_type}, is not a valid. "
                    'It must be one of "sobol", "halton", "LHS", and "OA-LHS".'
                )
                raise ValueError(message)

        assert isinstance(qmc_engine, scipy.stats.qmc.QMCEngine)

        forward_size = sample_id - qmc_engine.num_generated  # `sample_id` starts from 0.
        qmc_engine.fast_forward(forward_size)
        sample = qmc_engine.random(1)
        self._cached_qmc_engine = qmc_engine

        return sample

    def _find_sample_id(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> int:

        qmc_id = ""
        qmc_id += self._qmc_type
        qmc_id += str(search_space)
        # Sobol/Halton sequences without scrambling do not use seed.
        if not (self._qmc_type in ("sobol", "halton") and (self._scramble is False)):
            qmc_id += str(self._seed)
        hashed_qmc_id = hash(qmc_id)
        key_qmc_id = f"qmc ({hashed_qmc_id})'s last sample id"

        # TODO(kstoneriv3): Following try-except block assumes that the block is
        # an atomic transaction. This ensures that each sample_id is sampled at least once.
        try:
            sample_id = study._storage.get_study_system_attrs(study._study_id)[key_qmc_id]
            sample_id += 1
            study._storage.set_study_system_attr(study._study_id, key_qmc_id, sample_id)
        except KeyError:
            study._storage.set_study_system_attr(study._study_id, key_qmc_id, 0)
            sample_id = 0

        return sample_id

    def _is_engine_cached(self, d: int, sample_id: int) -> bool:

        if not isinstance(self._cached_qmc_engine, scipy.stats.qmc.QMCEngine):
            return False
        else:
            is_cached = True
            is_cached &= self._cached_qmc_engine.rng_seed == self._seed
            is_cached &= self._cached_qmc_engine.d == d
            is_cached &= self._cached_qmc_engine.num_generated <= sample_id

            # TODO(kstoneriv3): Maybe we should assume that
            # `_qmc_type` does not change for simplicity.
            if self._qmc_type == "sobol":
                is_cached &= self._cached_qmc_engine.__class__.__name__ == "Sobol"
            elif self._qmc_type == "halton":
                is_cached &= self._cached_qmc_engine.__class__.__name__ == "Halton"
            elif self._qmc_type == "LHS":  # Latin Hypercube Sampling
                is_cached &= self._cached_qmc_engine.__class__.__name__ == "LatinHypercube"
            elif self._qmc_type == "OA-LHS":  # Orthogonal array-based Latin hypercube sampling
                is_cached &= (
                    self._cached_qmc_engine.__class__.__name__ == "OrthogonalLatinHypercube"
                )
            else:
                message = (
                    f"The `qmc_type`, {self._qmc_type}, is not a valid. "
                    'It must be one of "sobol", "halton", "LHS", and "OA-LHS".'
                )
                raise ValueError(message)

            return is_cached
