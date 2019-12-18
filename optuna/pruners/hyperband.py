from optuna import logging
from optuna.pruners.base import BasePruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA

_logger = logging.get_logger(__name__)


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

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial noted as :math:`r`
            in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper. See the details for
            :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        min_early_stopping_rate_low:
            The start point of the minimum early stopping rate for ``SuccessiveHalvingPruner``.
        min_early_stopping_rate_high:
            The end point of the minimum early stopping rate for ``SuccessiveHalvingPruner``.
    """

    def __init__(
            self,
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate_low=0,
            min_early_stopping_rate_high=4
    ):
        # type: (int, int, int, int) -> None
        self._pruners = []  # type: List[SuccessiveHalvingPruner]
        self._reduction_factor = reduction_factor
        self._resource_budget = 0
        n_pruners = min_early_stopping_rate_high - min_early_stopping_rate_low + 1
        self._n_pruners = n_pruners
        self._bracket_resource_budgets = []  # type: List[int]

        _logger.debug('Hyperband has {} brackets'.format(self.n_pruners))

        for i in range(n_pruners):
            bracket_resource_budget = self._calc_bracket_resource_budget(i, n_pruners)
            self._resource_budget += bracket_resource_budget
            self._bracket_resource_budgets.append(bracket_resource_budget)

            # N.B. (crcrpar): `min_early_stopping_rate` has the information of `bracket_index`.
            min_early_stopping_rate = min_early_stopping_rate_low + i
            rung_key_prefix = 'bracket{}_'.format(i)

            _logger.debug(
                '{}th bracket has minimum early stopping rate of {}'.format(
                    i, min_early_stopping_rate))

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
                rung_key_prefix=rung_key_prefix
            )
            self._pruners.append(pruner)

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        i = self.get_bracket_id(study.study_name, trial.number)
        _logger.debug('{}th bracket is selected'.format(i))
        return self._pruners[i].prune(study, trial)

    def get_trial_pruner_auxiliary_data(self, study_name, trial_number):
        # type: (str, int) -> str

        bracket_index = self.get_bracket_id(study_name, trial_number)
        return '_{}'.format(bracket_index)

    def should_filter_trials(self):
        # type: () -> bool

        return True

    def _calc_bracket_resource_budget(self, pruner_index, n_pruners):
        # type: (int, int) -> int
        n = self._reduction_factor ** (n_pruners - 1)
        budget = n
        for i in range(pruner_index, n_pruners - 1):
            budget += n / 2
        return budget

    def get_bracket_id(self, study_name, trial_number):
        # type: (str, int) -> int
        """Computes the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_.
        """

        n = hash('{}_{}'.format(study_name, trial_number)) % self._resource_budget
        for i in range(self.n_pruners):
            n -= self._bracket_resource_budgets[i]
            if n < 0:
                return i

        raise RuntimeError

    @property
    def resource_budget(self):
        # type: () -> int
        """The total amount of resource.

        This value is a proxy for :math:`B` in the
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_.
        """

        return self._resource_budget

    @property
    def n_pruners(self):
        # type: () -> int
        """The number of pruners.

        The number of pruners is calculated by the following equation:
        :math:`
        \\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}\\_\\mathsf{high} -
        \\mathsf{min}\\_\\mathsf{early}\\_\\mathsf{stopping}\\_\\mathsf{rate}\\_\\mathsf{low} + 1`
        """

        return self._n_pruners
