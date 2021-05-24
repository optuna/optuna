from typing import Tuple

import optuna
from optuna._experimental import experimental
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


@experimental("2.8.0")
class MaxTrialsCallback:
    """Set a maximum number of trials before ending the study.

    While the :obj:`n_trials` argument of :obj:`optuna.optimize` sets the number of trials that
    will be run, you may want to continue running until you have a certain number of successfullly
    completed trials or stop the study when you have a certain number of trials that fail.
    This :obj:`MaxTrialsCallback` class allows you to set a maximum number of trials for a
    particular :class:`~optuna.trial.TrialState` before stopping the study.

    Example:

        .. testcode::

            import optuna
            from optuna.study import MaxTrialsCallback
            from optuna.trial import TrialState


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x ** 2


            study = optuna.create_study()
            study.optimize(
                objective,
                callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))],
            )

    Args:
        n_trials:
            The max number of trials. Must be set to an integer.
        states:
            Tuple of the :class:`~optuna.trial.TrialState` to be counted
            towards the max trials limit. Default value is :obj:`(TrialState.COMPLETE,)`.
    """

    def __init__(
        self, n_trials: int, states: Tuple[TrialState, ...] = (TrialState.COMPLETE,)
    ) -> None:
        self._n_trials = n_trials
        self._states = states

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        trials = study.get_trials(deepcopy=False, states=self._states)
        n_complete = len(trials)
        if n_complete >= self._n_trials:
            study.stop()


@experimental("2.8.0")
def RetryFailedTrialCallback(study, trial):
    """When a trail fails, this callback can be used with the :class:`optuna.storage` class to
    recreate the trial in :obj:`TrialState.WAITING` to queue up the trial to be run again.

    This is helpful in environments where trials may fail due to external conditions, such as
    being pre-empted by other processes.

    Usage:

        .. testcode::

            storage = optuna.storages.RDBStorage(
                args.optuna_storage,
                heartbeat_interval=60,
                grace_period=120,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            )

            study = optuna.create_study(
                storage=storage,
            )
    """

    def __init__(self, max_retry: int) -> None:
        self._max_retry = max_retry

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        failed_trials = (
            trial.user_attrs["failed_trials"] if "failed_trials" in trial.user_attrs else []
        )
        failed_trials.append(trial.number)
        if failed_trials.count(trial.number) > self._max_retry:
            return

        trial.user_attrs.update({"last_failed_trial": trial.number})
        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.WAITING,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
            )
        )
