import abc
import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from optuna import study
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna import distributions  # NOQA
    from optuna.trial import FrozenTrial  # NOQA

DEFAULT_STUDY_NAME_PREFIX = "no-name-"


class BaseStorage(object, metaclass=abc.ABCMeta):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    Storage classes abstract a backend database and provide library internal interfaces to
    read/write history of studies and trials.

    **Thread safety**

    Storage classes might be shared from multiple threads, and thus storage classes
    must be thread-safe.
    As one of the requirements of the thread-safety, storage classes must guarantee
    that the returned values, such as `FrozenTrial`s will not be directly modified
    by storage class.
    However, storage class can assume that return values are never modified by users.
    When users modify return values of storage classes, it might break the internal states
    of storage classes, which will result in undefined behaviors.

    **Ownership of RUNNING trials**

    Trials in finished states are not allowed to be modified.
    Trials in the WAITING state are not allowed to be modified except for the `state` field.
    Storage classes can assume that each RUNNING trial is modified from only one process.
    When users modify a RUNNING trial from multiple processes, it might lead to
    an inconsistent internal state, which will result in undefined behaviors.
    To use optuna with MPI or in other multi-process programs, users must make sure
    that the optuna interface is accessed from only one of the processes.
    Storage classes are not designed to provide inter-process communication functionalities.

    **Consistency models**

    Storage classes must support monotonic-reads consistency model, that is, if a
    process reads a data `X`, any successive reads on data `X` does not return
    older values.
    They must support read-your-writes, that is, if a process writes to data `X`,
    any successive reads on data `X` from the same process must read the written
    value or one of more recent values.

    **Stronger consistency requirements for special data**

    TODO(ytsmiling) Add load method to storage class implementations.

    Under multi-worker settings, storage classes are guaranteed to return the latest
    values of any attributes of `Study`, but not guaranteed the same thing for
    attributes of `Trial`.
    However, if `load(study_id)` method is called, any successive reads on the `state`
    attribute of `Trial` in the study are guaranteed to return the same or more recent
    values than the value at the time the `load` method called.
    Let `T` be a `Trial`.
    Let `P` be a process that last updated the `state` attribute of `T`.
    Then, any reads on any attributes of `T` are guaranteed to return the same or
    more recent values than any writes by `P` on the attribute before `P` updated
    the `state` attribute of `T`.
    The same applies for `user_attrs', 'system_attrs', 'intermediate_values` attributes,
    but future development may allow storage class users to explicitly skip the above
    properties for these attributes.

    **Data persistence**

    Storage classes do not guarantee that write operations are logged into a persistent
    storage even when write methods succeed.
    Thus, when process failure occurs, some writes might be lost.
    As exceptions, when a persistent storage is available, any writes on any attributes
    of `Study` and writes on `state` of `Trial` are guaranteed to be persistent.
    Additionally, any preceding writes on any attributes of `Trial` are guaranteed to
    be written into a persistent storage before writes on `state` of `Trial` succeed.
    The same applies for `user_attrs', 'system_attrs', 'intermediate_values` attributes,
    but future development may allow storage class users to explicitly skip the above
    properties for these attributes.
    """

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study(self, study_name: Optional[str] = None) -> int:
        """Create a new study with a given name.

        When no name is specified, storage class auto-generates the name.
        Study ID is unique among all current and deleted studies.

        Args:
            study_name:
                Name of a new study to create.

        Returns:
            Study ID, which is an unique id among studies, of the created study.

        Raises:
            :exc:`optuna.exceptions.DuplicatedStudyError`:
                If a study with the same ``study_name`` already exists.
        """
        # TODO(ytsmiling) Fix RDB storage implementation to ensure unique `study_id`.
        raise NotImplementedError

    @abc.abstractmethod
    def delete_study(self, study_id: int) -> None:
        """Delete a study specified by the study ID.

        Args:
            study_id:
                Study ID of a study to delete.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register a user-defined attribute to a study.

        This method overwrites an existing attribute.

        Args:
            study_id:
                Study ID of a study to update.
            key:
                Key of an attribute.
            value:
                Attributes' value.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register an optuna-internal attribute to a study.

        This method overwrites an existing attribute.

        Args:
            study_id:
                Study ID of a study to update.
            key:
                Key of an attribute.
            value:
                Attributes' value.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_direction(self, study_id: int, direction: study.StudyDirection) -> None:
        """Register a direction of optimization problem to a study.

        Args:
            study_id:
                Study ID of a study to update.
            direction:
                Either :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                :obj:`~optuna.study.StudyDirection.MINIMIZE`.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`ValueError`:
                If direction is already set and the passed ``direction`` is the opposite
                direction or :obj:`~optuna.study.StudyDirection.NOT_SET`.
        """
        raise NotImplementedError

    # Basic study access

    @abc.abstractmethod
    def get_study_id_from_name(self, study_name: str) -> int:
        """Read study ID of a study with the same name.

        Args:
            study_name:
                Name of a study to search the study id.

        Returns:
            Study ID of a study with the matching study name.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        """Read study ID of a study that a specified trial belongs to.

        Args:
            trial_id:
                Trial ID of a trial to search the study id of a study the trial belongs to.
        Returns:
            Study id of a trial with the given trial id.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_name_from_id(self, study_id: int) -> str:
        """Read study name of a study with a matching study ID.

        Args:
            study_id:
                Study ID of a study to search its name.
        Returns:
            Name of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_direction(self, study_id: int) -> study.StudyDirection:
        """Read whether a specified study maximizes or minimizes an objective.

        Args:
            study_id:
                Study ID of a study to read.
        Returns:
            Optimization direction of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read a dictionary of user-defined attributes of a specified study.

        Args:
            study_id:
                Study ID of a study to read.
        Returns:
            A dictionary of a user attributes of a study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read a dictionary of optuna-internal attributes of a specified study.

        Args:
            study_id:
                Study ID of a study to read.
        Returns:
            A dictionary of a system attributes of a study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_study_summaries(self) -> List[study.StudySummary]:
        """Read a list of :class:`~optuna.study.StudySummary` objects.

        Returns:
            A list of :class:`~optuna.study.StudySummary` objects.

        """
        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(
        self, study_id: int, template_trial: Optional["FrozenTrial"] = None
    ) -> int:
        """Create and add a new trial to a specified study.

        Trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                Study ID to add a new trial.
            template_trial:
                Fronzen trial with default user-attributes, system-attributes,
                intermediate-values, and a state.

        Returns:
            Trial ID of the created trial.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        """Update a state of a specified trial.

        This method succeeds only when trial is not already finished.

        Args:
            trial_id:
                Trial ID of a trial to update its state.
            state:
                New state of the trial.

        Returns:
            :obj:`True` if the state is successfully updated.
            :obj:`False` if the state is kept the same.
            The latter happens when this method tries to update the state of
            :obj:`~optuna.trial.TrialState.RUNNING` trial to
            :obj:`~optuna.trial.TrialState.RUNNING`.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: "distributions.BaseDistribution",
    ) -> bool:
        """Add a parameter to a specified trial.

        Args:
            trial_id:
                Trial ID of a trial to add a parameter.
            param_name:
                Name of a parameter to add.
            param_value_internal:
                Internal representation of a value a parameter to add.
            distribution:
                Sampled distribution of a parameter to add.

        Returns:
            Return :obj:`False` when the parameter is already set to the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_number_from_id(self, trial_id: int) -> int:
        """Read a trial number of a specified trial.

        Trial ID is a unique ID of a trial, while trial number is a unique and
        sequential ID of a trial within a study.

        Args:
            trial_id:
                Trial ID of a trial to read.

        Returns:
            Trial number of a trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        """Read a specified parameter of a trial.

        Args:
            trial_id:
                Trial ID of a trial to read.
            param_name:
                Name of a parameter to read.

        Returns:
            Internal representation of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
                If no such parameter exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_value(self, trial_id: int, value: float) -> None:
        """Set a return value of an objective function.

        This method overwrites existing trial value.

        Args:
            trial_id:
                Trial ID of a trial to set value.
            value:
                The return value of an objective.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> bool:
        """Report a value within an evaluation of an objective function.

        Args:
            trial_id:
                Trial ID of a trial to set intermediate value.
            step:
                Step of the trial (e.g., Epoch of neural network training).
            intermediate_value:
                Reported value within an evaluation of an objective function.

        Returns:
            Return :obj:`False` when the intermediate of the step already exists.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set a user-defined attribute to a specified trial.

        This method overwrites an existing attribute.

        Args:
            trial_id:
                Trial ID of a trial to set user-defined attribute.
            key:
                Key of an attribute to register.
            value:
                Value of the attribute. The value should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set an optuna-internal attribute to a specified trial.

        This method overwrites an existing attribute.

        Args:
            trial_id:
                Trial ID of a trial to set optuna-internal attribute.
            key:
                Key of an attribute to register.
            value:
                Value of the attribute. The value should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id: int) -> "FrozenTrial":
        """Read a trial using a trial ID.

        Args:
            trial_id:
                Trial ID of a trial to read.

        Returns:
            Trial with a matching trial ID.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List["FrozenTrial"]:
        """Read all trials in a specified study.

        Args:
            study_id:
                Study ID of a study to read trials from.
            deepcopy:
                Whether copy the list of trials before returning.
                Set :obj:`True` when you might update the list or elements of the list.

        Returns:
            A list of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:
        """Count the number of trials in a specified study.

        Args:
            study_id:
                Study ID of a study to count trials.
            state:
                :class:`~optuna.trial.TrialState` to filter trials.

        Returns:
            Number of the trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    def get_best_trial(self, study_id: int) -> "FrozenTrial":
        """Return a trial with the best value in the study.

        Args:
            study_id:
                Study ID to search the best trial.

        Returns:
            The trial with the best return value of the objective function
            among all finished tirals in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`RuntimeError`:
                If no trials have been completed.
        """
        all_trials = self.get_all_trials(study_id, deepcopy=False)
        all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError("No trials are completed yet.")

        if self.get_study_direction(study_id) == study.StudyDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: t.value)
        else:
            best_trial = min(all_trials, key=lambda t: t.value)

        return copy.deepcopy(best_trial)

    def get_trial_params(self, trial_id: int) -> Dict[str, Any]:
        """Read parameter dictionary of a specified trial.

        Args:
            trial_id:
                A trial ID of a trial to read parameters.

        Returns:
            A dictionary of a parameters consisting of parameter names and internal
            representations of the parameters' values.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read a user-defined attributes of a specified trial.

        Args:
            trial_id:
                A trial ID of a trial to read user-defined attributes.

        Returns:
            A dictionary of user-defined attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read an optuna-internal attributes of a specified trial.

        Args:
            trial_id:
                A trial ID of a trial to read optuna-internal attributes.

        Returns:
            A dictionary of optuna-internal attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).system_attrs

    def remove_session(self) -> None:
        """Clean up all connections to a database."""
        pass

    def check_trial_is_updatable(self, trial_id: int, trial_state: TrialState) -> None:
        """Check whether a trial state is updatable.

        Args:
            trial_id:
                Trial id of a trial to update.
                Only used for an error message.
            trial_state:
                Trial state to check.

        Raises:
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number)
            )
