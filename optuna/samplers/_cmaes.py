import copy
import math
import pickle
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

from cmaes import CMA
from cmaes import get_warm_start_mgd
from cmaes import SepCMA
import numpy as np

import optuna
from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


CmaClass = Union[CMA, SepCMA]


class CmaEsSampler(BaseSampler):
    """A sampler using `cmaes <https://github.com/CyberAgentAILab/cmaes>`_ as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.CmaEsSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x**2 + y


            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Please note that this sampler does not support CategoricalDistribution.
    However, :class:`~optuna.distributions.FloatDistribution` with ``step``,
    (:func:`~optuna.trial.Trial.suggest_float`) and
    :class:`~optuna.distributions.IntDistribution` (:func:`~optuna.trial.Trial.suggest_int`)
    are supported.

    If your search space contains categorical parameters, I recommend you
    to use :class:`~optuna.samplers.TPESampler` instead.
    Furthermore, there is room for performance improvements in parallel
    optimization settings. This sampler cannot use some trials for updating
    the parameters of multivariate normal distribution.

    For further information about CMA-ES algorithm, please refer to the following papers:

    - `N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
      <https://arxiv.org/abs/1604.00772>`_
    - `A. Auger and N. Hansen. A restart CMA evolution strategy with increasing population
      size. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005),
      pages 1769–1776. IEEE Press, 2005.
      <http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf>`_
    - `Raymond Ros, Nikolaus Hansen. A Simple Modification in CMA-ES Achieving Linear Time and
      Space Complexity. 10th International Conference on Parallel Problem Solving From Nature,
      Sep 2008, Dortmund, Germany. inria-00287367.
      <https://hal.inria.fr/inria-00287367/document>`_
    - `Masahiro Nomura, Shuhei Watanabe, Youhei Akimoto, Yoshihiko Ozaki, Masaki Onishi.
      Warm Starting CMA-ES for Hyperparameter Optimization, AAAI. 2021.
      <https://arxiv.org/abs/2012.06932>`_

    .. seealso::
        You can also use :class:`optuna.integration.PyCmaSampler` which is a sampler using cma
        library as the backend.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used. Note that ``x0`` is sampled uniformly
            within the search space domain for each restart if you specify ``restart_strategy``
            argument.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for CMA-ES.

        n_startup_trials:
            The independent sampling is used instead of the CMA-ES algorithm until the given number
            of trials finish in the same study.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CmaEsSampler` is determined by
            :func:`~optuna.samplers.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.

        restart_strategy:
            Strategy for restarting CMA-ES optimization when converges to a local minimum.
            If given :obj:`None`, CMA-ES will not restart (default).
            If given 'ipop', CMA-ES will restart with increasing population size.
            Please see also ``inc_popsize`` parameter.

            .. note::
                Added in v2.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.1.0.

        inc_popsize:
            Multiplier for increasing population size before each restart.
            This argument will be used when setting ``restart_strategy = 'ipop'``.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

            .. note::
                It is suggested to set this flag :obj:`False` when the
                :class:`~optuna.pruners.MedianPruner` is used. On the other hand, it is suggested
                to set this flag :obj:`True` when the :class:`~optuna.pruners.HyperbandPruner` is
                used. Please see `the benchmark result
                <https://github.com/optuna/optuna/pull/1229>`_ for the details.

        use_separable_cma:
            If this is :obj:`True`, the covariance matrix is constrained to be diagonal.
            Due to reduce the model complexity, the learning rate for the covariance matrix
            is increased. Consequently, this algorithm outperforms CMA-ES on separable functions.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

        source_trials:
            This option is for Warm Starting CMA-ES, a method to transfer prior knowledge on
            similar HPO tasks through the initialization of CMA-ES. This method estimates a
            promising distribution from ``source_trials`` and generates the parameter of
            multivariate gaussian distribution. Please note that it is prohibited to use
            ``x0``, ``sigma0``, or ``use_separable_cma`` argument together.

            .. note::
                Added in v2.6.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.6.0.

    """

    def __init__(
        self,
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: Optional[str] = None,
        inc_popsize: int = 2,
        use_separable_cma: bool = False,
        source_trials: Optional[List[FrozenTrial]] = None,
    ) -> None:
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._cma_rng = np.random.RandomState(seed)
        self._search_space = optuna.samplers.IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._restart_strategy = restart_strategy
        self._inc_popsize = inc_popsize
        self._use_separable_cma = use_separable_cma
        self._source_trials = source_trials

        if self._restart_strategy:
            warnings.warn(
                "`restart_strategy` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if self._consider_pruned_trials:
            warnings.warn(
                "`consider_pruned_trials` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if self._use_separable_cma:
            warnings.warn(
                "`use_separable_cma` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if self._source_trials is not None:
            warnings.warn(
                "`source_trials` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if source_trials is not None and (x0 is not None or sigma0 is not None):
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when "
                "x0 or sigma0 is specified."
            )

        # TODO(c-bata): Support WS-sep-CMA-ES.
        if source_trials is not None and use_separable_cma:
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES."
            )

        # TODO(c-bata): Support BIPOP-CMA-ES.
        if restart_strategy not in (
            "ipop",
            None,
        ):
            raise ValueError(
                "restart_strategy={} is unsupported. Please specify: 'ipop' or None.".format(
                    restart_strategy
                )
            )

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:

        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            _logger.info(
                "`CmaEsSampler` only supports two or more dimensional continuous "
                "search space. `{}` is used instead of `CmaEsSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            self._warn_independent_sampling = False
            return {}

        trans = _SearchSpaceTransform(search_space)

        optimizer, n_restarts = self._restore_optimizer(completed_trials)
        if optimizer is None:
            n_restarts = 0
            optimizer = self._init_optimizer(trans, study.direction)

        if self._restart_strategy is None:
            generation_attr_key = "cma:generation"  # For backward compatibility.
        else:
            generation_attr_key = "cma:restart_{}:generation".format(n_restarts)

        if optimizer.dim != len(trans.bounds):
            _logger.info(
                "`CmaEsSampler` does not support dynamic search space. "
                "`{}` is used instead of `CmaEsSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            self._warn_independent_sampling = False
            return {}

        # TODO(c-bata): Reduce the number of wasted trials during parallel optimization.
        # See https://github.com/optuna/optuna/pull/920#discussion_r385114002 for details.
        solution_trials = [
            t
            for t in completed_trials
            if optimizer.generation == t.system_attrs.get(generation_attr_key, -1)
        ]
        if len(solution_trials) >= optimizer.population_size:
            solutions: List[Tuple[np.ndarray, float]] = []
            for t in solution_trials[: optimizer.population_size]:
                assert t.value is not None, "completed trials must have a value"
                x = trans.transform(t.params)
                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value
                solutions.append((x, y))

            optimizer.tell(solutions)

            if self._restart_strategy == "ipop" and optimizer.should_stop():
                n_restarts += 1
                generation_attr_key = "cma:restart_{}:generation".format(n_restarts)
                popsize = optimizer.population_size * self._inc_popsize
                optimizer = self._init_optimizer(
                    trans, study.direction, population_size=popsize, randomize_start_point=True
                )

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = _split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        # Caution: optimizer should update its seed value.
        seed = self._cma_rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        params = optimizer.ask()

        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        study._storage.set_trial_system_attr(trial._trial_id, "cma:n_restarts", n_restarts)

        external_values = trans.untransform(params)

        return external_values

    def _restore_optimizer(
        self,
        completed_trials: "List[optuna.trial.FrozenTrial]",
    ) -> Tuple[Optional[CmaClass], int]:
        if not self._use_separable_cma:
            attr_key_optimizer = "cma:optimizer"
            attr_key_n_restarts = "cma:n_restarts"
        else:
            attr_key_optimizer = "sepcma:optimizer"
            attr_key_n_restarts = "sepcma:n_restarts"

        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            optimizer_attrs = {
                key: value
                for key, value in trial.system_attrs.items()
                if key.startswith(attr_key_optimizer)
            }
            if len(optimizer_attrs) == 0:
                continue

            if not self._use_separable_cma and "cma:optimizer" in optimizer_attrs:
                # Check "cma:optimizer" key for backward compatibility.
                optimizer_str = optimizer_attrs["cma:optimizer"]
            else:
                optimizer_str = _concat_optimizer_attrs(optimizer_attrs)

            n_restarts: int = trial.system_attrs.get(attr_key_n_restarts, 0)
            return pickle.loads(bytes.fromhex(optimizer_str)), n_restarts
        return None, 0

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: Optional[int] = None,
        randomize_start_point: bool = False,
    ) -> CmaClass:
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._source_trials is None:
            if randomize_start_point:
                mean = lower_bounds + (upper_bounds - lower_bounds) * self._cma_rng.rand(
                    n_dimension
                )
            elif self._x0 is None:
                mean = lower_bounds + (upper_bounds - lower_bounds) / 2
            else:
                # `self._x0` is external representations.
                mean = trans.transform(self._x0)

            if self._sigma0 is None:
                sigma0 = np.min((upper_bounds - lower_bounds) / 6)
            else:
                sigma0 = self._sigma0

            cov = None
        else:
            expected_states = [TrialState.COMPLETE]
            if self._consider_pruned_trials:
                expected_states.append(TrialState.PRUNED)

            # TODO(c-bata): Filter parameters by their values instead of checking search space.
            sign = 1 if direction == StudyDirection.MINIMIZE else -1
            source_solutions = [
                (trans.transform(t.params), sign * cast(float, t.value))
                for t in self._source_trials
                if t.state in expected_states
                and _is_compatible_search_space(trans, t.distributions)
            ]
            if len(source_solutions) == 0:
                raise ValueError("No compatible source_trials")

            # TODO(c-bata): Add options to change prior parameters (alpha and gamma).
            mean, sigma0, cov = get_warm_start_mgd(source_solutions)

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, _EPS)

        if self._use_separable_cma:
            return SepCMA(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                seed=self._cma_rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

        return CMA(
            mean=mean,
            sigma=sigma0,
            cov=cov,
            bounds=trans.bounds,
            seed=self._cma_rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `CmaEsSampler` "
            "(optimization performance may be degraded). "
            "`CmaEsSampler` does not support dynamic search space or `CategoricalDistribution`. "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `CmaEsSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: "optuna.Study") -> List[FrozenTrial]:
        complete_trials = []
        for t in study.get_trials(deepcopy=False):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:

        self._independent_sampler.after_trial(study, trial, state, values)


def _split_optimizer_str(optimizer_str: str) -> Dict[str, str]:
    optimizer_len = len(optimizer_str)
    attrs = {}
    for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
        start = i * _SYSTEM_ATTR_MAX_LENGTH
        end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
        attrs["cma:optimizer:{}".format(i)] = optimizer_str[start:end]
    return attrs


def _is_compatible_search_space(
    trans: _SearchSpaceTransform, search_space: Dict[str, BaseDistribution]
) -> bool:
    intersection_size = len(set(trans._search_space.keys()).intersection(search_space.keys()))
    return intersection_size == len(trans._search_space) == len(search_space)


def _concat_optimizer_attrs(optimizer_attrs: Dict[str, str]) -> str:
    return "".join(
        optimizer_attrs["cma:optimizer:{}".format(i)] for i in range(len(optimizer_attrs))
    )
