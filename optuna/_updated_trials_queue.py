from collections import deque
import threading
from typing import Any
from typing import Deque
from typing import Dict
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optuna import Study
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState


class UpdatedTrialsQueue(object):
    """A virtual queue of trials in the specified states.

    This class imitates a queue that trial is added when it goes into specified states.

    It is not supposed to be directly accessed by library users except to write user-defined
    samplers.

    Note that the ``states`` argument should consist of the same stage states.
    """

    def __init__(self, study: "Study", states: Tuple["TrialState", ...]) -> None:
        for state in states:
            if state.is_promotable_to(states[0]) or states[0].is_promotable_to(state):
                raise RuntimeError("The states should be in the same stage.")

        self._study = study
        self._states = states

        self._queue: Deque[int] = deque()
        self._watching_trial_indices: List[int] = []
        self._next_min_trial_index = 0

        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[Any, Any]:

        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:

        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _fetch_trials(self, deepcopy: bool) -> List["FrozenTrial"]:
        trials = self._study.get_trials(deepcopy=deepcopy)
        next_watching_trial_indices: List[int] = []

        for trial_index in self._watching_trial_indices:
            if trials[trial_index].state in self._states:
                self._queue.append(trial_index)
            elif trials[trial_index].state.is_promotable_to(self._states[0]):
                next_watching_trial_indices.append(trial_index)

        for trial_index in range(self._next_min_trial_index, len(trials)):
            trial = trials[trial_index]
            if trial.state in self._states:
                self._queue.append(trial_index)
            elif trial.state.is_promotable_to(self._states[0]):
                next_watching_trial_indices.append(trial_index)

        self._watching_trial_indices = next_watching_trial_indices
        self._next_min_trial_index = len(trials)

        return trials

    def get(self, deepcopy: bool = True) -> "FrozenTrial":
        """Returns an unseen trial whose state is as specified.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.

        Returns:
            An unseen :class:`~optuna.Trial` object with the specified state.

        Raises:
            IndexError:
                If no trial is in the queue.
        """

        with self._lock:
            trials = self._fetch_trials(deepcopy=deepcopy)
            while True:
                trial_index = self._queue.popleft()
                if trials[trial_index].state in self._states:
                    return trials[trial_index]

    def get_trials(self, deepcopy: bool = True) -> List["FrozenTrial"]:
        """Returns a list of unseen trials whose states are as specified.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.

        Returns:
            A list of unseen :class:`~optuna.Trial` objects with the specified state.
        """

        with self._lock:
            trials = self._fetch_trials(deepcopy=deepcopy)
            ret = []
            for trial_index in self._queue:
                if trials[trial_index].state in self._states:
                    ret.append(trials[trial_index])
            self._queue.clear()
            return ret
