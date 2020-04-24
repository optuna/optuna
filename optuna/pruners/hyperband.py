import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import warnings

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import logging
from optuna.pruners.base import BasePruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner
from optuna.samplers.base import BaseSampler
from optuna.trial import FrozenTrial

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
        of Hyperband and is automatically determined by ``max_resource`` and ``reduction_factor``
        as `The number of brackets = floor(log(max_resource) / log(reduction_factor)) + 1`. Please
        set ``reduction_factor`` so that the number of brackets is not too large　(about 4 ~ 6 in
        most use cases).　Please see Section 3.6 of the `original paper
        <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_ for the detail.

    Args:
        min_resource:
            A parameter for specifying the minimum resource allocated to a trial noted as :math:`r`
            in the paper.
            See the details for :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        max_resource:
            A parameter for specifying the maximum resource allocated to a trial noted as :math:`R`
            in the paper. This value represents and should match the maximum iteration steps (e.g.,
            the number of epochs for neural networks).
        reduction_factor:
            A parameter for specifying reduction factor of promotable trials noted as
            :math:`\\eta` in the paper. See the details for
            :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        n_brackets:

            .. deprecated:: 1.4.0
                This argument will be removed from :class:~optuna.pruners.HyperbandPruner. The
                number of brackets are automatically determined based on ``max_resource`` and
                ``reduction_factor``.

            The number of :class:`~optuna.pruners.SuccessiveHalvingPruner`\\ s (brackets).
            Defaults to :math:`4`.
        min_early_stopping_rate_low:
            A parameter for specifying the minimum early-stopping rate.
            This parameter is related to a parameter that is referred to as :math:`r` and used in
            `Asynchronous SuccessiveHalving paper <http://arxiv.org/abs/1810.05934>`_.
            The minimum early stopping rate for :math:`i` th bracket is :math:`i + s`.
    """

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: int = 80,
        reduction_factor: int = 3,
        n_brackets: Optional[int] = None,
        min_early_stopping_rate_low: int = 0,
    ) -> None:

        self._pruners = []  # type: List[SuccessiveHalvingPruner]
        self._reduction_factor = reduction_factor
        self._total_trial_allocation_budget = 0

        if n_brackets is None:
            # In the original paper http://www.jmlr.org/papers/volume18/16-558/16-558.pdf, the
            # inputs of Hyperband are `R`: max resource and `\eta`: reduction factor. The
            # number of brackets (this is referred as `s_{max} + 1` in the paper) is calculated
            # by s_{max} + 1 = \floor{\log_{\eta} (R)} + 1 in Algorithm 1 of the original paper.
            self._n_brackets = (
                math.floor(math.log2(max_resource) / math.log2(reduction_factor)) + 1
            )
        else:
            message = (
                "The argument of `n_brackets` is deprecated. "
                "The number of brackets is automatically determined by `max_resource` and "
                "`reduction_factor` as "
                "`n_brackets = floor(log(max_resource) / log(reduction_factor)) + 1`. "
                "Please specify `reduction_factor` appropriately."
            )
            warnings.warn(message, DeprecationWarning)
            _logger.warning(message)
            self._n_brackets = n_brackets

        self._trial_allocation_budgets = []  # type: List[int]

        _logger.debug("Hyperband has {} brackets".format(self._n_brackets))

        for i in range(self._n_brackets):
            trial_allocation_budget = self._calculate_trial_allocation_budget(i)
            self._total_trial_allocation_budget += trial_allocation_budget
            self._trial_allocation_budgets.append(trial_allocation_budget)

            # N.B. (crcrpar): `min_early_stopping_rate` has the information of `bracket_index`.
            min_early_stopping_rate = min_early_stopping_rate_low + i

            _logger.debug(
                "{}th bracket has minimum early stopping rate of {}".format(
                    i, min_early_stopping_rate
                )
            )

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
            )
            self._pruners.append(pruner)

    def prune(self, study: "optuna.study.Study", trial: FrozenTrial) -> bool:
        i = self._get_bracket_id(study, trial)
        _logger.debug("{}th bracket is selected".format(i))
        bracket_study = self._create_bracket_study(study, i)
        return self._pruners[i].prune(bracket_study, trial)

    def _calculate_trial_allocation_budget(self, pruner_index: int) -> int:
        """Computes the trial allocated budget for a bracket of ``pruner_index``.

        In the `original paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`, the
        number of trials per one bracket is referred as ``n`` in Algorithm 1. Since we do not know
        the total number of trials in the leaning scheme of Optuna, we calculate the ratio of the
        number of trials here instead.
        """

        s = self._n_brackets - 1 - pruner_index
        return self._n_brackets * (self._reduction_factor ** s) // (s + 1)

    def _get_bracket_id(self, study: "optuna.study.Study", trial: FrozenTrial) -> int:
        """Computes the index of bracket for a trial of ``trial_number``.

        The index of a bracket is noted as :math:`s` in
        `Hyperband paper <http://www.jmlr.org/papers/volume18/16-558/16-558.pdf>`_.
        """

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


class _HyperbandSampler(BaseSampler):
    def __init__(self, sampler: BaseSampler, hyperband_pruner: HyperbandPruner) -> None:
        self._sampler = sampler
        self._pruner = hyperband_pruner

    @property
    def sampler(self) -> BaseSampler:
        return self._sampler

    @sampler.setter
    def sampler(self, new_sampler: BaseSampler) -> None:
        self._sampler = new_sampler

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        study = self._pruner._create_bracket_study(
            study, self._pruner._get_bracket_id(study, trial)
        )
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        study = self._pruner._create_bracket_study(
            study, self._pruner._get_bracket_id(study, trial)
        )
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        study = self._pruner._create_bracket_study(
            study, self._pruner._get_bracket_id(study, trial)
        )
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)
