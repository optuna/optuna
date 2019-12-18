import abc

from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA


class BasePruner(object, metaclass=abc.ABCMeta):
    """Base class for pruners."""

    @abc.abstractmethod
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool
        """Judge whether the trial should be pruned at the given step.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            study:
                Study object of the target study.
            trial:
                FrozenTrial object of the target trial.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_pruner_auxiliary_data(self, study_name, trial_number):
        # type: (str, int) -> str
        """Return appropriate pruner's metadata for a trial."""

        raise NotImplementedError

    @abc.abstractmethod
    def should_filter_trials(self):
        # type: () -> bool
        """Returns whether the sampler can use all of :func:`~optuna.study.Study.trials`.

        This method tells :class:`~optuna.study.Study` whether it needs to filters out trials
        that have different ``pruner_metadata`` of :func:`optuna.trial.Trial.user_attrs`.
        One use-case of this method is :class:`~optuna.pruners.HyperbandPruner` that runs and
        manages multiple :class:`~optuna.pruners.SuccessiveHalvingPruner`\\s called ``bracket``\\s
        by indexing its brackets.
        Therefore, the sampler of the study should use the history of trials that is monitored by
        the pruner of the same index.
        """

        raise NotImplementedError
