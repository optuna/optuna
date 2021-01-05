import abc
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


DEFAULT_STUDY_NAME_PREFIX = "no-name-"


class BaseStorage(object, metaclass=abc.ABCMeta):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    A storage class abstracts a backend database and provides library internal interfaces to
    read/write histories of studies and trials.

    **Thread safety**

    A storage class can be shared among multiple threads, and must therefore be thread-safe.
    It must guarantee that return values such as `FrozenTrial`s are never modified.
    A storage class can assume that return values are never modified by its user.
    When a user modifies a return value from a storage class, the internal state of the storage
    may become inconsistent. Consequences are undefined.

    **Ownership of RUNNING trials**

    Trials in finished states are not allowed to be modified.
    Trials in the WAITING state are not allowed to be modified except for the `state` field.
    A storage class can assume that each RUNNING trial is only modified from a single process.
    When a user modifies a RUNNING trial from multiple processes, the internal state of the storage
    may become inconsistent. Consequences are undefined.
    A storage class is not intended for inter-process communication.
    Consequently, users using optuna with MPI or other multi-process programs must make sure that
    only one process is used to access the optuna interface.

    **Consistency models**

    A storage class must support the monotonic-reads consistency model, that is, if a
    process reads data `X`, any successive reads on data `X` cannot return older values.
    It must support read-your-writes, that is, if a process writes to data `X`,
    any successive reads on data `X` from the same process must read the written
    value or one of the more recent values.

    **Stronger consistency requirements for special data**

    Under a multi-worker setting, a storage class must return the latest values of any attributes
    of a study, not necessarily for the attributes of a `Trial`.
    However, if the `read_trials_from_remote_storage(study_id)` method is called, any successive
    reads on the `state` attribute of a `Trial` are guaranteed to return the same or more recent
    values than the value at the time of the call to the
    `read_trials_from_remote_storage(study_id)` method.
    Let `T` be a `Trial`.
    Let `P` be the process that last updated the `state` attribute of `T`.
    Then, any reads on any attributes of `T` are guaranteed to return the same or
    more recent values than any writes by `P` on the attribute before `P` updated
    the `state` attribute of `T`.
    The same applies for `user_attrs', 'system_attrs' and 'intermediate_values` attributes.

    .. note::

        These attribute behaviors may become user customizable in the future.

    **Data persistence**

    A storage class does not guarantee that write operations are logged into a persistent
    storage, even when write methods succeed.
    Thus, when process failure occurs, some writes might be lost.
    As exceptions, when a persistent storage is available, any writes on any attributes
    of `Study` and writes on `state` of `Trial` are guaranteed to be persistent.
    Additionally, any preceding writes on any attributes of `Trial` are guaranteed to
    be written into a persistent storage before writes on `state` of `Trial` succeed.
    The same applies for `user_attrs', 'system_attrs' and 'intermediate_values` attributes.

    .. note::

        These attribute behaviors may become user customizable in the future.
    """

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study(self, study_name: Optional[str] = None) -> int:
        """Create a new study from a name.

        If no name is specified, the storage class generates a name.
        The returned study ID is unique among all current and deleted studies.

        Args:
            study_name:
                Name of the new study to create.

        Returns:
            ID of the created study.

        Raises:
            :exc:`optuna.exceptions.DuplicatedStudyError`:
                If a study with the same ``study_name`` already exists.
        """
        # TODO(ytsmiling) Fix RDB storage implementation to ensure unique `study_id`.
        raise NotImplementedError

    @abc.abstractmethod
    def delete_study(self, study_id: int) -> None:
        """Delete a study.

        Args:
            study_id:
                ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register a user-defined attribute to a study.

        This method overwrites any existing attribute.

        Args:
            study_id:
                ID of the study.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register an optuna-internal attribute to a study.

        This method overwrites any existing attribute.

        Args:
            study_id:
                ID of the study.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:
        """Register optimization problem directions to a study.

        Args:
            study_id:
                ID of the study.
            directions:
                A sequence of direction whose element is either
                :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                :obj:`~optuna.study.StudyDirection.MINIMIZE`.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`ValueError`:
                If the directions are already set and the each coordinate of passed ``directions``
                is the opposite direction or :obj:`~optuna.study.StudyDirection.NOT_SET`.
        """
        raise NotImplementedError

    # Basic study access

    @abc.abstractmethod
    def get_study_id_from_name(self, study_name: str) -> int:
        """Read the ID of a study.

        Args:
            study_name:
                Name of the study.

        Returns:
            ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_name`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        """Read the ID of a study to which a trial belongs.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            ID of the study.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_name_from_id(self, study_id: int) -> str:
        """Read the study name of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Name of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        """Read whether a study maximizes or minimizes an objective.

        Args:
            study_id:
                ID of a study.

        Returns:
            Optimization directions list of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the user-defined attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the user attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the optuna-internal attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the optuna-internal attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_study_summaries(self) -> List[StudySummary]:
        """Read a list of :class:`~optuna.study.StudySummary` objects.

        Returns:
            A list of :class:`~optuna.study.StudySummary` objects.

        """
        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        """Create and add a new trial to a study.

        The returned trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                ID of the study.
            template_trial:
                Template :class:`~optuna.trial.FronzenTrial` with default user-attributes,
                system-attributes, intermediate-values, and a state.

        Returns:
            ID of the created trial.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        """Update the state of a trial.

        Args:
            trial_id:
                ID of the trial.
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
        distribution: BaseDistribution,
    ) -> None:
        """Set a parameter to a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.
            param_value_internal:
                Internal representation of the parameter value.
            distribution:
                Sampled distribution of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        """Read the trial id of a trial.

        Args:
            study_id:
                ID of the study.
            trial_number:
                Number of the trial.

        Returns:
            ID of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``study_id`` and ``trial_number`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_number_from_id(self, trial_id: int) -> int:
        """Read the trial number of a trial.

        .. note::

            The trial number is only unique within a study, and is sequential.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Number of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        """Read the parameter of a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.

        Returns:
            Internal representation of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
                If no such parameter exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_values(self, trial_id: int, values: Sequence[float]) -> None:
        """Set return values of an objective function.

        This method overwrites any existing trial values.

        Args:
            trial_id:
                ID of the trial.
            values:
                Values of the objective function.

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
    ) -> None:
        """Report an intermediate value of an objective function.

        This method overwrites any existing intermediate value associated with the given step.

        Args:
            trial_id:
                ID of the trial.
            step:
                Step of the trial (e.g., the epoch when training a neural network).
            intermediate_value:
                Intermediate value corresponding to the step.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set a user-defined attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set an optuna-internal attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id: int) -> FrozenTrial:
        """Read a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Trial with a matching trial ID.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Tuple[TrialState, ...]] = None,
    ) -> List[FrozenTrial]:
        """Read all trials in a study.

        Args:
            study_id:
                ID of the study.
            deepcopy:
                Whether to copy the list of trials before returning.
                Set to :obj:`True` if you intend to update the list or elements of the list.
            states:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            List of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    def get_n_trials(
        self, study_id: int, state: Optional[Union[Tuple[TrialState, ...], TrialState]] = None
    ) -> int:
        """Count the number of trials in a study.

        Args:
            study_id:
                ID of the study.
            state:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            Number of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        # TODO(hvy): Align the name and the behavior or the `state` parameter with
        # `get_all_trials`'s `states`.
        if isinstance(state, TrialState):
            state = (state,)
        return len(self.get_all_trials(study_id, deepcopy=False, states=state))

    def get_best_trial(self, study_id: int) -> FrozenTrial:
        """Return the trial with the best value in a study.

        This method is valid only during single-objective optimization.

        Args:
            study_id:
                ID of the study.

        Returns:
            The trial with the best objective value among all finished trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`RuntimeError`:
                If the study has more than one direction.
            :exc:`ValueError`:
                If no trials have been completed.
        """
        all_trials = self.get_all_trials(study_id, deepcopy=False)
        all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError("No trials are completed yet.")

        directions = self.get_study_directions(study_id)
        if len(directions) > 1:
            raise RuntimeError(
                "Best trial can be obtained only for single-objective optimization."
            )
        direction = directions[0]

        if direction == StudyDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: cast(float, t.value))
        else:
            best_trial = min(all_trials, key=lambda t: cast(float, t.value))

        return best_trial

    def get_trial_params(self, trial_id: int) -> Dict[str, Any]:
        """Read the parameter dictionary of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary of a parameters. Keys are parameter names and values are internal
            representations of the parameter values.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read the user-defined attributes of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary with the user-defined attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read the optuna-internal attributes of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary with the optuna-internal attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).system_attrs

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        """Make an internal cache of trials up-to-date.

        Args:
            study_id:
                ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    def remove_session(self) -> None:
        """Clean up all connections to a database."""
        pass

    def check_trial_is_updatable(self, trial_id: int, trial_state: TrialState) -> None:
        """Check whether a trial state is updatable.

        Args:
            trial_id:
                ID of the trial.
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
