from __future__ import annotations

import binascii
import copy
import math
import threading
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import optuna
from optuna import logging
from optuna._warnings import optuna_warn
from optuna.pruners._base import BasePruner
from optuna.pruners._successive_halving import SuccessiveHalvingPruner
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from collections.abc import Container

    from optuna.storages import BaseStorage
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = logging.get_logger(__name__)


class HyperbandPruner(BasePruner):
    """Pruner using Hyperband.

    This pruner makes pruning decisions within each bracket. To also use bracket-specific
    history for sampling, set the study's sampler to :class:`~optuna.samplers.HyperbandSampler`
    and specify an integer ``max_resource``. With an unwrapped sampler, the pruner emits a
    warning during pruning and continues to make bracket-specific pruning decisions.

    As SuccessiveHalving (SHA) requires the number of configurations
    :math:`n` as its hyperparameter.  For a given finite budget :math:`B`,
    all the configurations have the resources of :math:`B \\over n` on average.
    As you can see, there will be a trade-off of :math:`B` and :math:`B \\over n`.
    `Hyperband <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__ attacks this trade-off
    by trying different :math:`n` values for a fixed budget.

    .. note::
        * In the Hyperband paper, the counterpart of :class:`~optuna.samplers.RandomSampler`
          is used.
        * :class:`~optuna.samplers.HyperbandSampler` uses :class:`~optuna.samplers.TPESampler`
          by default.
        * The `original benchmark results
          <https://github.com/optuna/optuna/pull/828#issuecomment-575457360>`__ cover the earlier
          integration of ``HyperbandPruner`` with both samplers.

    .. note::
        When using the default :class:`~optuna.samplers.HyperbandSampler`, consider setting larger
        ``n_trials`` or ``timeout`` to give each bracket's TPE sampler time to finish startup.

        Each bracket uses its own :class:`~optuna.samplers.TPESampler`, which needs
        ``n_startup_trials`` completed or pruned trials (10 by default) before using their results.
        With four brackets, at least :math:`4 \\times 10` trials are consumed for startup.
        Trials are distributed unevenly across brackets, so some brackets may require a larger
        overall trial budget to finish startup.

    .. note::
        Hyperband has several :class:`~optuna.pruners.SuccessiveHalvingPruner`\\ s. Each
        :class:`~optuna.pruners.SuccessiveHalvingPruner` is referred to as "bracket" in the
        original paper. The number of brackets is an important factor to control the early
        stopping behavior of Hyperband and is automatically determined by ``min_resource``,
        ``max_resource`` and ``reduction_factor`` as
        :math:`\\mathrm{The\\ number\\ of\\ brackets} =
        \\mathrm{floor}(\\log_{\\texttt{reduction}\\_\\texttt{factor}}
        (\\frac{\\texttt{max}\\_\\texttt{resource}}{\\texttt{min}\\_\\texttt{resource}})) + 1`.
        Please set ``reduction_factor`` so that the number of brackets is not too large (about 4 –
        6 in most use cases). Please see Section 3.6 of the `original paper
        <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__ for the detail.

    .. note::
        ``HyperbandPruner`` computes bracket ID for each trial with a
        function taking ``study_name`` of :class:`~optuna.study.Study` and
        :attr:`~optuna.trial.Trial.number`. Please specify ``study_name``
        to make the pruning algorithm reproducible.

    Example:

        Maximize validation accuracy with Hyperband pruning and bracket-specific sampling.

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)
            n_train_iter = 100


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.HyperbandSampler(),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1, max_resource=n_train_iter, reduction_factor=3
                ),
            )
            study.optimize(objective, n_trials=20)

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial noted as :math:`r`
            in the paper. A smaller :math:`r` will give a result faster, but a larger
            :math:`r` will give a better guarantee of successful judging between configurations.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        max_resource:
            A parameter for specifying the maximum resource allocated to a trial. :math:`R` in the
            paper corresponds to ``max_resource / min_resource``. This value represents and should
            match the maximum iteration steps (e.g., the number of epochs for neural networks).
            When this argument is "auto", the maximum resource is estimated according to the
            completed trials. The default value of this argument is "auto".

            .. note::
                With "auto", the maximum resource will be the largest step reported by
                :meth:`~optuna.trial.Trial.report` in the first, or one of the first if trained in
                parallel, completed trial. No trials will be pruned until the maximum resource is
                determined.

            .. note::
                If the step of the last intermediate value may change with each trial, please
                manually specify the maximum possible step to ``max_resource``.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        bootstrap_count:
            Parameter specifying the number of trials required in a rung before any trial can be
            promoted. Incompatible with ``max_resource`` is ``"auto"``.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
    """

    _initialization_lock = threading.Lock()

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._pruners: list[SuccessiveHalvingPruner] = []
        self._bootstrap_count = bootstrap_count
        self._total_trial_allocation_budget = 0
        self._trial_allocation_budgets: list[int] = []
        self._n_brackets: int | None = None

        if not isinstance(self._max_resource, int) and self._max_resource != "auto":
            raise ValueError(
                "The 'max_resource' should be integer or 'auto'. "
                f"But max_resource = {self._max_resource}"
            )

        if self._bootstrap_count > 0 and self._max_resource == "auto":
            raise ValueError(
                "bootstrap_count > 0 and max_resource == 'auto' "
                f"are mutually incompatible, bootstrap_count is {self._bootstrap_count}"
            )

        if isinstance(max_resource, int):
            self._initialize_brackets(max_resource)

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        # Subclasses can define their own sampler pairing.
        if type(self) is HyperbandPruner and not isinstance(
            study.sampler, optuna.samplers.HyperbandSampler
        ):
            optuna_warn(
                "HyperbandPruner is being used without HyperbandSampler. "
                "Wrap the sampler with HyperbandSampler and set an integer max_resource "
                "to use only the current bracket's history when proposing new trials."
            )
        if not self._pruners:
            self._try_initialization(study)
            if not self._pruners:
                return False

        bracket_id = self._get_bracket_id(study, trial)
        _logger.debug(f"{bracket_id}th bracket is selected")
        bracket_study = self._create_bracket_study(study, bracket_id)
        return self._pruners[bracket_id].prune(bracket_study, trial)

    def _try_initialization(self, study: Study) -> None:
        max_resource = self._max_resource
        if not isinstance(max_resource, int):
            trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            n_steps = [t.last_step for t in trials if t.last_step is not None]
            if not n_steps:
                return

            max_resource = max(n_steps) + 1

        with self._initialization_lock:
            if not self._pruners:
                self._initialize_brackets(max_resource)

    def _initialize_brackets(self, max_resource: int) -> None:
        # Algorithm 1 of the Hyperband paper uses R = max_resource / min_resource.
        resource_ratio = max_resource / self._min_resource
        n_brackets = math.floor(math.log(resource_ratio, self._reduction_factor)) + 1
        _logger.debug(f"Hyperband has {n_brackets} brackets")

        # Use the paper's trial counts as relative weights for assigning trials to brackets.
        budgets = [
            math.ceil(n_brackets * self._reduction_factor**s / (s + 1))
            for s in reversed(range(n_brackets))
        ]
        pruners = [
            SuccessiveHalvingPruner(
                min_resource=self._min_resource,
                reduction_factor=self._reduction_factor,
                min_early_stopping_rate=bracket_id,
                bootstrap_count=self._bootstrap_count,
            )
            for bracket_id in range(n_brackets)
        ]
        self._n_brackets = n_brackets
        self._trial_allocation_budgets = budgets
        self._total_trial_allocation_budget = sum(budgets)
        self._pruners = pruners
        self._max_resource = max_resource

    def _get_bracket_id(self, study: Study, trial: FrozenTrial) -> int:
        """Compute the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`__.
        """

        if not self._pruners:
            # A pruner restored from an older pickle may still need to build its brackets.
            if isinstance(self._max_resource, int):
                self._try_initialization(study)
            if not self._pruners:
                return 0

        assert self._n_brackets is not None
        n = (
            binascii.crc32(f"{study.study_name}_{trial.number}".encode())
            % self._total_trial_allocation_budget
        )
        for bracket_id in range(self._n_brackets):
            n -= self._trial_allocation_budgets[bracket_id]
            if n < 0:
                return bracket_id

        assert False, "This line should be unreachable."

    def _create_bracket_study(
        self,
        study: Study,
        bracket_id: int,
    ) -> Study:
        bracket_study = copy.copy(study)
        bracket_study._storage = cast("BaseStorage", _BracketStorage(study, self, bracket_id))
        bracket_study.stop = study.stop  # type: ignore[method-assign]
        return bracket_study


class _BracketStorage:
    def __init__(self, study: Study, pruner: HyperbandPruner, bracket_id: int) -> None:
        self._study = study
        self._storage = study._storage
        self._pruner = pruner
        self._bracket_id = bracket_id

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_storage"), name)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        trials = self._storage.get_all_trials(study_id, deepcopy=False, states=states)
        if study_id == self._study._study_id:
            trials = [
                trial
                for trial in trials
                if self._pruner._get_bracket_id(self._study, trial) == self._bracket_id
            ]
        return copy.deepcopy(trials) if deepcopy else trials
