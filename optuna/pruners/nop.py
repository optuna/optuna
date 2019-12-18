from optuna.pruners import BasePruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna import structs  # NOQA
    from optuna.study import Study  # NOQA


class NopPruner(BasePruner):
    """Pruner which never prunes trials.

    Example:

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import NopPruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=NopPruner())
            >>> study.optimize(objective)
    """

    def prune(self, study, trial):
        # type: (Study, structs.FrozenTrial) -> bool

        return False

    def get_trial_pruner_auxiliary_data(self, study_name, trial_number):
        # type: (str, int) -> str

        return ''

    def should_filter_trials(self):
        # type: () -> bool

        return False
