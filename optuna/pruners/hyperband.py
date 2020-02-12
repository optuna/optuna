import optuna
from optuna._experimental import experimental
from optuna import logging
from optuna.pruners.base import BasePruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

    from optuna import structs  # NOQA

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
    Note that this implementation does not take as inputs the maximum amount of resource to
    a single SHA noted as :math:`R` in the paper.

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
        at least :math:`4 \\times 10` pruners are consumed for startup.

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial noted as :math:`r`
            in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper. See the details for
            :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        n_brackets:
            The number of :class:`~optuna.pruners.SuccessiveHalvingPruner`\\ s (brackets).
            Defaults to :math`4`. See
            https://github.com/optuna/optuna/pull/809#discussion_r361363897.
        min_early_stopping_rate_low:
            A parameter for specifying the minimum early-stopping rate.
            This parameter is related to a parameter that is referred to as :math:`r` and used in
            `Asynchronous SuccessiveHalving paper <http://arxiv.org/abs/1810.05934>`_.
            The minimum early stopping rate for :math:`i` th bracket is :math:`i + s`.
    """

    def __init__(
            self,
            min_resource=1,
            reduction_factor=3,
            n_brackets=4,
            min_early_stopping_rate_low=0
    ):
        # type: (int, int, int, int) -> None

        self._pruners = []  # type: List[SuccessiveHalvingPruner]
        self._reduction_factor = reduction_factor
        self._resource_budget = 0
        self._n_brackets = n_brackets
        self._bracket_resource_budgets = []  # type: List[int]

        _logger.debug('Hyperband has {} brackets'.format(self._n_brackets))

        for i in range(n_brackets):
            bracket_resource_budget = self._calc_bracket_resource_budget(i, n_brackets)
            self._resource_budget += bracket_resource_budget
            self._bracket_resource_budgets.append(bracket_resource_budget)

            # N.B. (crcrpar): `min_early_stopping_rate` has the information of `bracket_index`.
            min_early_stopping_rate = min_early_stopping_rate_low + i

            _logger.debug(
                '{}th bracket has minimum early stopping rate of {}'.format(
                    i, min_early_stopping_rate))

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
            )
            self._pruners.append(pruner)

    def prune(self, study, trial):
        # type: (optuna.study.Study, structs.FrozenTrial) -> bool

        i = self._get_bracket_id(study, trial)
        _logger.debug('{}th bracket is selected'.format(i))
        bracket_study = self._create_bracket_study(study, i)
        return self._pruners[i].prune(bracket_study, trial)

    # TODO(crcrpar): Improve resource computation/allocation algorithm.
    def _calc_bracket_resource_budget(self, pruner_index, n_brackets):
        # type: (int, int) -> int

        n = self._reduction_factor ** (n_brackets - 1)
        return n + (n / 2) * (n_brackets - 1 - pruner_index)

    def _get_bracket_id(self, study, trial):
        # type: (optuna.study.Study, structs.FrozenTrial) -> int
        """Computes the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_.
        """

        n = hash('{}_{}'.format(study.study_name, trial.number)) % self._resource_budget
        for i in range(self._n_brackets):
            n -= self._bracket_resource_budgets[i]
            if n < 0:
                return i

        assert False, 'This line should be unreachable.'

    def _create_bracket_study(self, study, bracket_index):
        # type: (optuna.study.Study, int) -> optuna.study.Study

        # This class is assumed to be passed to
        # `SuccessiveHalvingPruner.prune` in which `get_trials`,
        # `direction`, and `storage` are used.
        # But for safety, prohibit the other attributes explicitly.
        class _BracketStudy(optuna.study.Study):

            _VALID_ATTRS = (
                'get_trials', 'direction', '_storage', '_study_id',
                'pruner', 'study_name', '_bracket_id', 'sampler'
            )

            def __init__(self, study, bracket_id):
                # type: (optuna.study.Study, int) -> None

                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=study.pruner
                )
                self._bracket_id = bracket_id

            def get_trials(self, deepcopy=True):
                # type: (bool) -> List[structs.FrozenTrial]

                trials = super().get_trials(deepcopy=deepcopy)
                pruner = self.pruner
                assert isinstance(pruner, HyperbandPruner)
                return [
                    t for t in trials
                    if pruner._get_bracket_id(self, t) == self._bracket_id
                ]

            def __getattribute__(self, attr_name):  # type: ignore
                if attr_name not in _BracketStudy._VALID_ATTRS:
                    raise AttributeError(
                        "_BracketStudy does not have attribute of '{}'".format(attr_name))
                else:
                    return object.__getattribute__(self, attr_name)

        return _BracketStudy(study, bracket_index)
