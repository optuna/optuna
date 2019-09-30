import abc
import six

from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna import distributions  # NOQA

DEFAULT_STUDY_NAME_PREFIX = 'no-name-'


@six.add_metaclass(abc.ABCMeta)
class BaseStorage(object):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    Storage classes abstract a backend database and provide library internal interfaces to
    read/write history of studies and trials.
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
        # type: (int, structs.StudyDirection) -> None

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
        # type: (int) -> structs.StudyDirection

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
        # type: () -> List[structs.StudySummary]

        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[structs.FrozenTrial]) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> None

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
        # type: (int) -> structs.FrozenTrial

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        raise NotImplementedError

    @abc.abstractmethod
    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        raise NotImplementedError

    def get_best_trial(self, study_id):
        # type: (int) -> structs.FrozenTrial

        all_trials = self.get_all_trials(study_id)
        all_trials = [t for t in all_trials if t.state is structs.TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError('No trials are completed yet.')

        if self.get_study_direction(study_id) == structs.StudyDirection.MAXIMIZE:
            return max(all_trials, key=lambda t: t.value)
        return min(all_trials, key=lambda t: t.value)

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
        # type: (int, structs.TrialState) -> None

        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number))
