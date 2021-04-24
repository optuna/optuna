from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy

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



@experimental("2.x.0")  # TODO(kstoneriv3)
class LatinHypercubeSampler(BaseSampler):
    # TODO(kstoneriv3): update the docstring
    """A Quasi Monte Carlo Sampler that generates low-discrepancy sequences.

    Quasi Monte Carlo (QMC) sequences are designed to have lower discrepancies than
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
        If your search space contains categorical parameters, it samples the catagorical
        parameters by its `independent_sampler` without using QMC algorithm.

    Args:
        qmc_type:
            The type of QMC sequence to be sampled. This must be one of
            `"halton"` and `"sobol"`. Default is `"halton"`.

            .. note:
                Sobol sequence is designed to have low-discrepancy property when the number of
                samples is :math:`n=2^m` for each positive integer :math:`m`. When it is possible
                to pre-specify the number of trials suggested by `QMCSampler`, it is recommended
                that the number of trials should be set as power of two.

        scramble:
            In cases ``qmc_type`` is `"halton"` or `"sobol"`, if this option is :obj:`True`,
            scrambling (randomization) is applied to the QMC sequences.

        seed:
            A seed for `QMCSampler`. When the ``qmc_type`` is `"halton"` or `"sobol"`,
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
            If ``qmc_type`` is not one of 'halton' an 'sobol`.

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
        n: int,
        *,
        seed: Optional[int] = None,
        search_space: Optional[Dict[str, BaseDistribution]] = None,
        add_noise: bool = True,
        independent_sampler: Optional[BaseSampler] = None,
        warn_incomplete_reseeding: bool = True,
    ) -> None:

        self._n = n
        self._seed = seed or numpy.random.MT19937().random_raw()
        self._add_noise = add_noise
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._cached_lhs = None
        # TODO(kstoneriv3): make sure that search_space is either None or valid search space.
        # also make sure that it is OrderedDict
        self._initial_search_space = search_space
        self._warn_incomplete_reseeding = warn_incomplete_reseeding
        self._should_stop = False

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()

        # We must not reseed the `self._seed` like below. Otherwise, workers will have different
        # seed under multiprocess execution because reseed_rng is called when forking process.
        # self._seed = numpy.random.MT19937().random_raw()
        if self._warn_incomplete_reseeding:
            self._log_incomplete_reseeding()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        if self._initial_search_space is not None:
            return self._initial_search_space

        past_trials = study._storage.get_all_trials(
            study._study_id, states=_SUGGESTED_STATES, deepcopy=False
        )

        # The initial trial is sampled by the independent sampler.
        if len(past_trials) == 0:
            return {}
        # If an initial trial was already made,
        # construct search_space of this sampler from the initial trial.
        else:
            first_trial = min(past_trials, key=lambda t: t.number)
            self._initial_search_space = self._infer_initial_search_space(first_trial)
            return self._initial_search_space

    def _infer_initial_search_space(self, trial: FrozenTrial) -> Dict[str, BaseDistribution]:

        return trial.distributions

    @staticmethod
    def _log_incomplete_reseeding() -> None:
        _logger.warning(
            "The seed of QMC seqeunce is not reseeded and only the seed of `independent_sampler` "
            "is reseeded. This is to ensure that each workers samples from the same QMC sequence "
            "in the parallel and/or distributed environment."
            "You can suppress this warning by setting `warn_reseeding` "
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

        sample = self._sample_lhs(study, search_space)
        trans = _SearchSpaceTransform(search_space)
        sample = trans.untransform(sample)
        return sample

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:

        self._independent_sampler.after_trial(study, trial, state, values)
        if self._should_stop:  # TODO(kstoneriv3): Consider using information on `state`
            study.stop()


    def _sample_lhs(
        self, study: Study, search_space: Dict[str, BaseDistribution]
    ) -> numpy.ndarray:

        sample_id = self._find_sample_id(study, search_space)
        d = sum([
            len(dist.choices) if isinstance(dist, distributions.CategoricalDistribution) else 1
            for dist in search_space.values()
        ])

        # TODO(kstoneriv3): Consider using information on `state`
        # TODO(kstoneriv3): Not sure if it properly stops in distributed environment.
        if sample_id == self._n - 1:
            self._should_stop = True

        # Use cached `lhs` (Latin hypercube samples) or construct a new one.
        if not self._is_lhs_cached(d, sample_id):
            self._precompute_lhs(d, search_space)

        sample = self._cached_lhs[sample_id, :]
        return sample

    def _find_sample_id(self, study: Study, search_space: Dict[str, BaseDistribution]) -> int:

        lhs_id = ""
        lhs_id += str(search_space)
        lhs_id += str(self._n)
        lhs_id += str(self._seed)
        hashed_lhs_id = hash(lhs_id)
        key_lhs_id = f"lhs ({hashed_lhs_id})'s last sample id"

        # TODO(kstoneriv3): Following try-except block assumes that the block is
        # an atomic transaction. Without this assumption, current implementation
        # only ensures that each `sample_id` is sampled at least once.
        try:
            sample_id = study._storage.get_study_system_attrs(study._study_id)[key_lhs_id]
            sample_id += 1
            study._storage.set_study_system_attr(study._study_id, key_lhs_id, sample_id)
        except KeyError:
            study._storage.set_study_system_attr(study._study_id, key_lhs_id, 0)
            sample_id = 0

        return sample_id

    def _is_lhs_cached(self, d: int, sample_id: int) -> bool:

        if isinstance(self._cached_lhs, numpy.ndarray):
            is_cached = True
            is_cached &= self._cached_lhs_seed == self._seed
            is_cached &= self._cached_lhs.shape[0] == self._n
            is_cached &= self._cached_lhs.shape[1] == d
            return is_cached
        else:
            return False

    def _precompute_lhs(self, d: int, search_space: Dict[str, BaseDistribution]) -> None:

        rng = numpy.random.RandomState(self._seed)
        samples = numpy.zeros([self._n, d])
        bound_idx = 0
        for param_name, dist in search_space.items():
            if isinstance(dist, distributions.CategoricalDistribution):
                c = len(dist.choices)
                choices = numpy.arange(self._n) % c
                rng.shuffle(choices)
                samples[range(self._n), bound_idx + choices] = 1
                bound_idx += c
            else:
                trans = search_space = {param_name: dist}
                trans = _SearchSpaceTransform(search_space)
                low, high = trans.bounds[:, 0], trans.bounds[:, 1]
                # Enforce a unique point per grid
                perm = rng.permutation(self._n)
                if self._add_noise:
                    perm = perm + rng.uniform(low=0, high=1, size=self._n)
                else:
                    perm = perm + 0.5
                samples[:, bound_idx] = low + (high - low) * perm / self._n
                bound_idx += 1
        
        self._cached_lhs_seed = self._seed
        self._cached_lhs = samples




