import enum


class TrialState(enum.Enum):
    """State of a :class:`~optuna.trial.Trial`.

    Attributes:
        RUNNING:
            The :class:`~optuna.trial.Trial` is running.
        COMPLETE:
            The :class:`~optuna.trial.Trial` has been finished without any error.
        PRUNED:
            The :class:`~optuna.trial.Trial` has been pruned with
            :class:`~optuna.exceptions.TrialPruned`.
        FAIL:
            The :class:`~optuna.trial.Trial` has failed due to an uncaught error.
    """

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4

    def __repr__(self) -> str:

        return str(self)

    def is_finished(self) -> bool:

        return self != TrialState.RUNNING and self != TrialState.WAITING

    def is_promotable_to(self, state: "TrialState") -> bool:
        """Returns whether the self's state is promotable to the argument's state."""

        if self == TrialState.WAITING:
            return state == TrialState.RUNNING or state.is_finished()
        elif self == TrialState.RUNNING:
            return state.is_finished()
        else:
            return False
