from typing import Any
from typing import Container
from typing import Dict
from typing import List
from typing import Optional

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class MaxTrialsCallback:
    """Set a maximum number of trials before ending the study.

    While the ``n_trials`` argument of :meth:`optuna.study.Study.optimize` sets the number of
    trials that will be run, you may want to continue running until you have a certain number of
    successfully completed trials or stop the study when you have a certain number of trials that
    fail. This ``MaxTrialsCallback`` class allows you to set a maximum number of trials for a
    particular :class:`~optuna.trial.TrialState` before stopping the study.

    Example:

        .. testcode::

            import optuna
            from optuna.study import MaxTrialsCallback
            from optuna.trial import TrialState


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x**2


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
            towards the max trials limit. Default value is ``(TrialState.COMPLETE,)``.
            If :obj:`None`, count all states.
    """

    def __init__(
        self, n_trials: int, states: Optional[Container[TrialState]] = (TrialState.COMPLETE,)
    ) -> None:
        self._n_trials = n_trials
        self._states = states

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        trials = study.get_trials(deepcopy=False, states=self._states)
        n_complete = len(trials)
        if n_complete >= self._n_trials:
            study.stop()


@experimental_class("2.8.0")
class RetryFailedTrialCallback:
    """Retry a failed trial up to a maximum number of times.

    When a trial fails, this callback can be used with a class in :mod:`optuna.storages` to
    recreate the trial in ``TrialState.WAITING`` to queue up the trial to be run again.

    The failed trial can be identified by the
    :func:`~optuna.storages.RetryFailedTrialCallback.retried_trial_number` function.
    Even if repetitive failure occurs (a retried trial fails again),
    this method returns the number of the original trial.
    To get a full list including the numbers of the retried trials as well as their original trial,
    call the :func:`~optuna.storages.RetryFailedTrialCallback.retry_history` function.

    This callback is helpful in environments where trials may fail due to external conditions,
    such as being preempted by other processes.

    Usage:

        .. testcode::

            import optuna
            from optuna.storages import RetryFailedTrialCallback

            storage = optuna.storages.RDBStorage(
                url="sqlite:///:memory:",
                heartbeat_interval=60,
                grace_period=120,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            )

            study = optuna.create_study(
                storage=storage,
            )

    .. seealso::
        See :class:`~optuna.storages.RDBStorage`.

    Args:
        max_retry:
            The max number of times a trial can be retried. Must be set to :obj:`None` or an
            integer. If set to the default value of :obj:`None` will retry indefinitely.
            If set to an integer, will only retry that many times.
        inherit_intermediate_values:
            Option to inherit `trial.intermediate_values` reported by
            :func:`optuna.trial.Trial.report` from the failed trial. Default is :obj:`False`.
    """

    def __init__(
        self, max_retry: Optional[int] = None, inherit_intermediate_values: bool = False
    ) -> None:
        self._max_retry = max_retry
        self._inherit_intermediate_values = inherit_intermediate_values

    def __call__(self, study: "optuna.study.Study", trial: FrozenTrial) -> None:
        system_attrs: Dict[str, Any] = {
            "failed_trial": trial.number,
            "retry_history": [],
            **trial.system_attrs,
        }
        system_attrs["retry_history"].append(trial.number)
        if self._max_retry is not None:
            if self._max_retry < len(system_attrs["retry_history"]):
                return

        study.add_trial(
            optuna.create_trial(
                state=optuna.trial.TrialState.WAITING,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=system_attrs,
                intermediate_values=(
                    trial.intermediate_values if self._inherit_intermediate_values else None
                ),
            )
        )

    @staticmethod
    @experimental_func("2.8.0")
    def retried_trial_number(trial: FrozenTrial) -> Optional[int]:
        """Return the number of the original trial being retried.

        Args:
            trial:
                The trial object.

        Returns:
            The number of the first failed trial. If not retry of a previous trial,
            returns :obj:`None`.
        """

        return trial.system_attrs.get("failed_trial", None)

    @staticmethod
    @experimental_func("3.0.0")
    def retry_history(trial: FrozenTrial) -> List[int]:
        """Return the list of retried trial numbers with respect to the specified trial.

        Args:
            trial:
                The trial object.

        Returns:
            A list of trial numbers in ascending order of the series of retried trials.
            The first item of the list indicates the original trial which is identical
            to the :func:`~optuna.storages.RetryFailedTrialCallback.retried_trial_number`,
            and the last item is the one right before the specified trial in the retry series.
            If the specified trial is not a retry of any trial, returns an empty list.
        """
        return trial.system_attrs.get("retry_history", [])
