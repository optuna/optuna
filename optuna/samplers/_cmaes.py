import copy
import math
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import warnings

from cmaes import CMA
import numpy as np

import optuna
from optuna import logging
from optuna._study_direction import StudyDirection
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


class CmaEsSampler(BaseSampler):
    """A Sampler using CMA-ES algorithm.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.CmaEsSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -1, 1)
                y = trial.suggest_int("y", -1, 1)
                return x ** 2 + y


            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Please note that this sampler does not support CategoricalDistribution.
    If your search space contains categorical parameters, I recommend you
    to use :class:`~optuna.samplers.TPESampler` instead.
    Furthermore, there is room for performance improvements in parallel
    optimization settings. This sampler cannot use some trials for updating
    the parameters of multivariate normal distribution.

    For further information about CMA-ES algorithm and its restarting strategy
    algorithm, please refer to the following papers:

    - `N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
      <https://arxiv.org/abs/1604.00772>`_
    - `A. Auger and N. Hansen. A restart CMA evolution strategy with increasing population
      size. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005),
      pages 1769–1776. IEEE Press, 2005.
      <http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf>`_

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

    Raises:
        ValueError:
            If ``restart_strategy`` is not 'ipop' or :obj:`None`.
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
                    optuna.distributions.UniformDistribution,
                    optuna.distributions.LogUniformDistribution,
                    optuna.distributions.DiscreteUniformDistribution,
                    optuna.distributions.IntUniformDistribution,
                    optuna.distributions.IntLogUniformDistribution,
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
            optimizer = self._init_optimizer(trans)

        if self._restart_strategy is None:
            generation_attr_key = "cma:generation"  # for backward compatibility
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
                    trans, population_size=popsize, randomize_start_point=True
                )

            # Store optimizer
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = _split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        # Caution: optimizer should update its seed value
        seed = self._cma_rng.randint(1, 2 ** 16) + trial.number
        optimizer._rng = np.random.RandomState(seed)
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
    ) -> Tuple[Optional[CMA], int]:
        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            optimizer_attrs = {
                key: value
                for key, value in trial.system_attrs.items()
                if key.startswith("cma:optimizer")
            }
            if len(optimizer_attrs) == 0:
                continue

            # Check "cma:optimizer" key for backward compatibility.
            optimizer_str = optimizer_attrs.get("cma:optimizer", None)
            if optimizer_str is None:
                optimizer_str = _concat_optimizer_attrs(optimizer_attrs)

            n_restarts: int = trial.system_attrs.get("cma:n_restarts", 0)
            return pickle.loads(bytes.fromhex(optimizer_str)), n_restarts
        return None, 0

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        population_size: Optional[int] = None,
        randomize_start_point: bool = False,
    ) -> CMA:

        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if randomize_start_point:
            mean = lower_bounds + (upper_bounds - lower_bounds) * self._cma_rng.rand(n_dimension)
        elif self._x0 is None:
            mean = lower_bounds + (upper_bounds - lower_bounds) / 2
        else:
            # `self._x0` is external representations.
            mean = trans.transform(self._x0)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, _EPS)
        return CMA(
            mean=mean,
            sigma=sigma0,
            bounds=trans.bounds,
            seed=self._cma_rng.randint(1, 2 ** 31 - 2),
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


def _split_optimizer_str(optimizer_str: str) -> Dict[str, str]:
    optimizer_len = len(optimizer_str)
    attrs = {}
    for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
        start = i * _SYSTEM_ATTR_MAX_LENGTH
        end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
        attrs["cma:optimizer:{}".format(i)] = optimizer_str[start:end]
    return attrs


def _concat_optimizer_attrs(optimizer_attrs: Dict[str, str]) -> str:
    return "".join(
        optimizer_attrs["cma:optimizer:{}".format(i)] for i in range(len(optimizer_attrs))
    )
