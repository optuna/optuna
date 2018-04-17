import abc
import numpy as np
import six
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Tuple  # NOQA

from pfnopt import distributions  # NOQA
from pfnopt import study_summary  # NOQA
from pfnopt import trial


@six.add_metaclass(abc.ABCMeta)
class BaseStorage(object):

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study_id(self):
        # type: () -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_id_from_uuid(self, study_uuid):
        # type: (str) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_uuid_from_id(self, study_id):
        # type: (int) -> str

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_system_attrs(self, study_id, system_attrs):
        # type: (int, study_summary.StudySystemAttributes) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_system_attrs(self, study_id):
        # type: (int) -> study_summary.StudySystemAttributes

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        raise NotImplementedError

    @abc.abstractmethod
    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_study_summaries(self):
        # type: () -> List[study_summary.StudySummary]

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_param_distribution(self, trial_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None

        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def set_trial_state(self, trial_id, state):
        # type: (int, trial.State) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_param(self, trial_id, param_name, param_value_in_internal_repr):
        # type: (int, str, float) -> None

        # TODO(Akiba): float? how about categorical?
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_system_attrs(self, trial_id, system_attrs):
        # type: (int, trial.SystemAttributes) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id):
        # type: (int) -> trial.Trial

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id):
        # type: (int) -> List[trial.Trial]

        raise NotImplementedError

    # Trial access utility

    def get_best_trial(self, study_id):
        # type: (int) -> trial.Trial

        all_trials = self.get_all_trials(study_id)
        all_trials = [t for t in all_trials if t.state is trial.State.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError('No trials are completed yet')

        return min(all_trials, key=lambda t: t.value)

    def get_trial_params(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).params

    def get_trial_system_attrs(self, trial_id):
        # type: (int) -> trial.SystemAttributes

        return self.get_trial(trial_id).system_attrs

    # Methods for the TPE sampler

    def get_trial_param_result_pairs(self, study_id, param_name):
        # type: (int, str) -> List[Tuple[float, float]]

        # Be careful: this method returns param values in internal representation
        all_trials = self.get_all_trials(study_id)

        return [
            (t.params_in_internal_repr[param_name], t.value)
            for t in all_trials
            if param_name in t.params and t.state is trial.State.COMPLETE
            # TODO(Akiba): We also want to use pruned results
        ]

    # Methods for the median pruner

    def get_best_intermediate_result_over_steps(self, trial_id):
        # type: (int) -> float

        return min(self.get_trial(trial_id).intermediate_values.values())

    def get_median_intermediate_result_over_trials(self, study_id, step):
        # type: (int, int) -> float

        all_trials = self.get_all_trials(study_id)

        return float(np.median([
            t.intermediate_values[step] for t in all_trials
            if step in t.intermediate_values
        ]))

    def remove_session(self):
        # type: () -> None

        pass
