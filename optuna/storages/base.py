import abc
import copy

from optuna import study
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna import distributions  # NOQA
    from optuna.trial import FrozenTrial  # NOQA

DEFAULT_STUDY_NAME_PREFIX = "no-name-"


class BaseStorage(object, metaclass=abc.ABCMeta):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    Storage classes abstract a backend database and provide library internal interfaces to
    read/write history of studies and trials.

    Storage classes might be shared from multiple threads, and thus storage classes
    must be thread-safe.
    However, storage class can assume that return values are never modified by users.
    When users modify return values of storage classes, it might break the internal states
    of storage classes, which will result in undefined behaviors.

    Storage classes must support monotonic-reads consistency model, that is, if a
    process reads a data `X`, any successive reads on data `X` does not return
    older values.
    They must virtually support read-your-writes, that is, if a process writes to
    data `X`, any successive reads on data `X` from the same process must read
    the written value or one of more recent values.

    Under multi-worker settings, storage classes are guaranteed to return the latest
    values of any attributes of `Study`, but not guaranteed the same thing for
    attributes of `Trial`.
    However, if `load(study_id)` method is called, any successive reads on `state` and
    `system_attrs` attributes of `Trial` in the study are guaranteed to return the
    same or more recent values than the value at the time the `load` method called.
    Let `T` be a `Trial`.
    Let `P` be a process that last updated the `state` or `system_attr` of `T`.
    Then, any reads on any attributes of `T` are guaranteed to return the same or
    more recent values than any writes by `P` on the attribute before `P` updated
    the `state` or `system_attr` of `T`.

    Storage classes do not guarantee that write operations are logged into a persistent
    storage even when write methods succeed.
    Thus, when process failure occurs, some writes might be lost.
    As exceptions, when a persistent storage is available, any writes on any attributes
    of `Study` and writes on `state` and `system_attr` of `Trial` are guaranteed to be
    persistent.
    Additionally, any preceding writes on any attributes of `Trial` are guaranteed to
    be written into a persistent storage before writes on `state` or `system_attr` of
    `Trial` succeed.
    """

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study(self, study_name=None):
        # type: (Optional[str]) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def delete_study(self, study_id):
        # type: (int) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_direction(self, study_id, direction):
        # type: (int, study.StudyDirection) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    # Basic study access

    @abc.abstractmethod
    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_direction(self, study_id):
        # type: (int) -> study.StudyDirection

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_study_summaries(self):
        # type: () -> List[study.StudySummary]

        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[FrozenTrial]) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_state(self, trial_id, state):
        # type: (int, TrialState) -> bool

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id):
        # type: (int) -> FrozenTrial

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id, deepcopy=True):
        # type: (int, bool) -> List[FrozenTrial]

        raise NotImplementedError

    @abc.abstractmethod
    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[TrialState]) -> int

        raise NotImplementedError

    def get_best_trial(self, study_id):
        # type: (int) -> FrozenTrial

        all_trials = self.get_all_trials(study_id, deepcopy=False)
        all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError("No trials are completed yet.")

        if self.get_study_direction(study_id) == study.StudyDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: t.value)
        else:
            best_trial = min(all_trials, key=lambda t: t.value)

        return copy.deepcopy(best_trial)

    def get_trial_params(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).system_attrs

    def remove_session(self):
        # type: () -> None

        pass

    def check_trial_is_updatable(self, trial_id, trial_state):
        # type: (int, TrialState) -> None

        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number)
            )
