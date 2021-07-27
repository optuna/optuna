import copy

import optuna
from optuna._experimental import experimental


@experimental("2.9.0")
def fail_stale_trials(study: "optuna.Study") -> None:
    """Fail stale trials and run their failure callbacks.

    The running trials whose heartbeat has not been updated for a long time will be failed,
    that is, those states will be changed to :obj:`~optuna.trial.TrialState.FAIL`.

    .. seealso::

        See :class:`~optuna.storages.RDBStorage`.

    Args:
        study:
            Study holding the trials to check.
    """
    storage = study._storage

    if not storage.is_heartbeat_enabled():
        return

    failed_trial_ids = storage.fail_stale_trials(study._study_id)
    failed_trial_callback = storage.get_failed_trial_callback()
    if failed_trial_callback is not None:
        for trial_id in failed_trial_ids:
            failed_trial = copy.deepcopy(storage.get_trial(trial_id))
            failed_trial_callback(study, failed_trial)
