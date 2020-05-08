import math
from typing import List
from typing import Optional
from typing import Union
import warnings

import optuna
from optuna._experimental import experimental
from optuna import logging
from optuna.pruners.base import BasePruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_logger = logging.get_logger(__name__)


@experimental("1.1.0")
class HyperbandPruner(BasePruner):
    """Pruner using Hyperband.

    As SuccessiveHalving (SHA) requires the number of configurations
    :math:`n` as its hyperparameter.  For a given finite budget :math:`B`,
    all the configurations have the resources of :math:`B \\over n` on average.
    As you can see, there will be a trade-off of :math:`B` and :math:`B \\over n`.
    `Hyperband <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_ attacks this trade-off
    by trying different :math:`n` values for a fixed budget.

    .. note::
        * In the Hyperband paper, the counterpart of :class:`~optuna.samplers.RandomSampler`
          is used.
        * Optuna uses :class:`~optuna.samplers.TPESampler` by default.
        * `The benchmark result
          <https://github.com/optuna/optuna/pull/828#issuecomment-575457360>`_
          shows that :class:`optuna.pruners.HyperbandPruner` supports both samplers.

    .. note::
        If you use ``HyperbandPruner`` with :class:`~optuna.samplers.TPESampler`,
        it's recommended to consider to set larger ``n_trials`` or ``timeout`` to make full use of
        the characteristics of :class:`~optuna.samplers.TPESampler`
        because :class:`~optuna.samplers.TPESampler` uses some (by default, :math:`10`)
        :class:`~optuna.trial.Trial`\\ s for its startup.

        As Hyperband runs multiple :class:`~optuna.pruners.SuccessiveHalvingPruner` and collect
        trials based on the current :class:`~optuna.trial.Trial`\\ 's bracket ID, each bracket
        needs to observe more than :math:`10` :class:`~optuna.trial.Trial`\\ s
        for :class:`~optuna.samplers.TPESampler` to adapt its search space.

        Thus, for example, if ``HyperbandPruner`` has :math:`4` pruners in it,
        at least :math:`4 \\times 10` trials are consumed for startup.

    .. note::
        Hyperband has several :class:`~optuna.pruners.SuccessiveHalvingPruner`. Each
        :class:`~optuna.pruners.SuccessiveHalvingPruner` is referred as "bracket" in the original
        paper. The number of brackets is an important factor to control the early stopping behavior
        of Hyperband and is automatically determined by ``min_resource``, ``max_resource`` and
        ``reduction_factor`` as
        `The number of brackets = floor(log_{reduction_factor}(max_resource / min_resource)) + 1`.
        Please set ``reduction_factor`` so that the number of brackets is not too large　(about 4 ~
        6 in most use cases).　Please see Section 3.6 of the `original paper
        <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_ for the detail.

    Example:

        We minimize an objective function with Hyperband pruning algorithm.

        .. testsetup::

            import numpy as np
            from sklearn.model_selection import train_test_split

            np.random.seed(seed=0)
            X = np.random.randn(200).reshape(-1, 1)
            y = np.where(X[:, 0] < 0.5, 0, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            classes = np.unique(y)

        .. testcode::

            import optuna
            from sklearn.linear_model import SGDClassifier

            n_train_iter = 100

            def objective(trial):
                alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_test, y_test)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                return clf.score(X_test, y_test)

            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=n_train_iter,
                    reduction_factor=3
                )
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
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        n_brackets:

            .. deprecated:: 1.4.0
                This argument will be removed from :class:`~optuna.pruners.HyperbandPruner`. The
                number of brackets are automatically determined based on ``min_resource``,
                ``max_resource`` and ``reduction_factor``.

            The number of :class:`~optuna.pruners.SuccessiveHalvingPruner`\\ s (brackets).
            Defaults to :math:`4`.
        min_early_stopping_rate_low:

            .. deprecated:: 1.4.0
                This argument will be removed from :class:`~optuna.pruners.HyperbandPruner`.

            A parameter for specifying the minimum early-stopping rate.
            This parameter is related to a parameter that is referred to as :math:`s` and used in
            `Asynchronous SuccessiveHalving paper <http://arxiv.org/abs/1810.05934>`_.
            The minimum early stopping rate for :math:`i` th bracket is :math:`i + s`.
    """

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: Union[str, int] = "auto",
        reduction_factor: int = 3,
        n_brackets: Optional[int] = None,
        min_early_stopping_rate_low: Optional[int] = None,
    ) -> None:

        self._min_resource = min_resource
        self._max_resource = max_resource
        self._reduction_factor = reduction_factor
        self._n_brackets = n_brackets
        self._min_early_stopping_rate_low = min_early_stopping_rate_low
        self._pruners = []  # type: List[SuccessiveHalvingPruner]
        self._total_trial_allocation_budget = 0
        self._trial_allocation_budgets = []  # type: List[int]

        if not isinstance(self._max_resource, int) and self._max_resource != "auto":
            raise ValueError(
                "The 'max_resource' should be integer or 'auto'. "
                "But max_resource = {}".format(self._max_resource)
            )

        if n_brackets is not None:
            message = (
                "The argument of `n_brackets` is deprecated. "
                "The number of brackets is automatically determined by `min_resource`, "
                "`max_resource` and `reduction_factor` as "
                "`n_brackets = floor(log_{reduction_factor}(max_resource / min_resource)) + 1`. "
                "Please specify `reduction_factor` appropriately."
            )
            warnings.warn(message, DeprecationWarning)
            _logger.warning(message)

        if min_early_stopping_rate_low is not None:
            message = (
                "The argument of `min_early_stopping_rate_low` is deprecated. "
                "Please specify `min_resource` appropriately."
            )
            warnings.warn(message, DeprecationWarning)
            _logger.warning(message)

    def prune(self, study: "optuna.study.Study", trial: FrozenTrial) -> bool:
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                return False

        i = self._get_bracket_id(study, trial)
        _logger.debug("{}th bracket is selected".format(i))
        bracket_study = self._create_bracket_study(study, i)
        return self._pruners[i].prune(bracket_study, trial)

    def _try_initialization(self, study: "optuna.study.Study") -> None:
        if self._max_resource == "auto":
            trials = study.get_trials(deepcopy=False)
            n_steps = [
                t.last_step
                for t in trials
                if t.state == TrialState.COMPLETE and t.last_step is not None
            ]

            if not n_steps:
                return

            self._max_resource = max(n_steps) + 1

        assert isinstance(self._max_resource, int)

        if self._n_brackets is None:
            # In the original paper http://www.jmlr.org/papers/volume18/16-558/16-558.pdf, the
            # inputs of Hyperband are `R`: max resource and `\eta`: reduction factor. The
            # number of brackets (this is referred as `s_{max} + 1` in the paper) is calculated
            # by s_{max} + 1 = \floor{\log_{\eta} (R)} + 1 in Algorithm 1 of the original paper.
            self._n_brackets = (
                math.floor(math.log2(self._max_resource) / math.log2(self._reduction_factor)) + 1
            )

        _logger.debug("Hyperband has {} brackets".format(self._n_brackets))

        for i in range(self._n_brackets):
            trial_allocation_budget = self._calculate_trial_allocation_budget(i)
            self._total_trial_allocation_budget += trial_allocation_budget
            self._trial_allocation_budgets.append(trial_allocation_budget)

            if self._min_early_stopping_rate_low is None:
                min_early_stopping_rate = i
            else:
                min_early_stopping_rate = self._min_early_stopping_rate_low + i

            _logger.debug(
                "{}th bracket has minimum early stopping rate of {}".format(
                    i, min_early_stopping_rate
                )
            )

            pruner = SuccessiveHalvingPruner(
                min_resource=self._min_resource,
                reduction_factor=self._reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
            )
            self._pruners.append(pruner)

    def _calculate_trial_allocation_budget(self, pruner_index: int) -> int:
        """Compute the trial allocated budget for a bracket of ``pruner_index``.

        In the `original paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`, the
        number of trials per one bracket is referred as ``n`` in Algorithm 1. Since we do not know
        the total number of trials in the leaning scheme of Optuna, we calculate the ratio of the
        number of trials here instead.
        """

        assert self._n_brackets is not None
        s = self._n_brackets - 1 - pruner_index
        return math.ceil(self._n_brackets * (self._reduction_factor ** s) / (s + 1))

    def _get_bracket_id(self, study: "optuna.study.Study", trial: FrozenTrial) -> int:
        """Compute the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_.
        """

        if len(self._pruners) == 0:
            return 0

        assert self._n_brackets is not None
        n = (
            hash("{}_{}".format(study.study_name, trial.number))
            % self._total_trial_allocation_budget
        )
        for i in range(self._n_brackets):
            n -= self._trial_allocation_budgets[i]
            if n < 0:
                return i

        assert False, "This line should be unreachable."

    def _create_bracket_study(
        self, study: "optuna.study.Study", bracket_index: int
    ) -> "optuna.study.Study":
        # This class is assumed to be passed to
        # `SuccessiveHalvingPruner.prune` in which `get_trials`,
        # `direction`, and `storage` are used.
        # But for safety, prohibit the other attributes explicitly.
        class _BracketStudy(optuna.study.Study):

            _VALID_ATTRS = (
                "get_trials",
                "direction",
                "_storage",
                "_study_id",
                "pruner",
                "study_name",
                "_bracket_id",
                "sampler",
            )

            def __init__(self, study: "optuna.study.Study", bracket_id: int) -> None:
                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=study.pruner,
                )
                self._bracket_id = bracket_id

            def get_trials(self, deepcopy: bool = True) -> List[FrozenTrial]:
                trials = super().get_trials(deepcopy=deepcopy)
                pruner = self.pruner
                assert isinstance(pruner, HyperbandPruner)
                return [t for t in trials if pruner._get_bracket_id(self, t) == self._bracket_id]

            def __getattribute__(self, attr_name):  # type: ignore
                if attr_name not in _BracketStudy._VALID_ATTRS:
                    raise AttributeError(
                        "_BracketStudy does not have attribute of '{}'".format(attr_name)
                    )
                else:
                    return object.__getattribute__(self, attr_name)

        return _BracketStudy(study, bracket_index)
