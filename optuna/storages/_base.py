from __future__ import annotations

import abc
from collections.abc import Container, Sequence
from typing import Any, cast

from optuna._typing import JSONSerializable
from optuna.distributions import BaseDistribution
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState


DEFAULT_STUDY_NAME_PREFIX = "no-name-"


class BaseStorage(abc.ABC):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    This class abstracts a backend database and provides internal interfaces to
    read/write histories of studies and trials.

    A storage class implementing this class must meet the following requirements.

    **Thread safety**

    A storage class instance can be shared among multiple threads, and must therefore be
    thread-safe. It must guarantee that a data instance read from the storage must not be modified
    by subsequent writes. For example, `FrozenTrial` instance returned by `get_trial`
    should not be updated by the subsequent `set_trial_xxx`. This is usually achieved by replacing
    the old data with a copy on `set_trial_xxx`.

    A storage class can also assume that a data instance returned are never modified by its user.
    When a user modifies a return value from a storage class, the internal state of the storage
    may become inconsistent. Consequences are undefined.

    **Ownership of RUNNING trials**

    Trials in finished states are not allowed to be modified.
    Trials in the WAITING state are not allowed to be modified except for the `state` field.
    """

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        """Create a new study from a name.

        If no name is specified, the storage class generates a name.
        The returned study ID is unique among all current and deleted studies.

        Args:
            directions:
                 A sequence of direction whose element is either
                 :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                 :obj:`~optuna.study.StudyDirection.MINIMIZE`.
            study_name:
                Name of the new study to create.

        Returns:
            ID of the created study.

        Raises:
            :exc:`optuna.exceptions.DuplicatedStudyError`:
                If a study with the same ``study_name`` already exists.
        """
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
    def set_study_system_attr(self, study_id: int, key: str, value: JSONSerializable) -> None:
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
    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
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
    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
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
    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
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
    def get_all_studies(self) -> list[FrozenStudy]:
        """Read a list of :class:`~optuna.study.FrozenStudy` objects.

        Returns:
            A list of :class:`~optuna.study.FrozenStudy` objects, sorted by ``study_id``.

        """
        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        """Create and add a new trial to a study.

        The returned trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                ID of the study.
            template_trial:
                Template :class:`~optuna.trial.FrozenTrial` with default user-attributes,
                system-attributes, intermediate-values, and a state.

        Returns:
            ID of the created trial.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
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
        """Read the trial ID of a trial.

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
        trials = self.get_all_trials(study_id, deepcopy=False)
        if len(trials) <= trial_number:
            raise KeyError(
                f"No trial with trial number {trial_number} exists in study with study_id {study_id}."
            )
        return trials[trial_number]._trial_id

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
        return self.get_trial(trial_id).number

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
        """
        return cast(float, self.get_trial(trial_id).params[param_name])

    @abc.abstractmethod
    def set_trial_state(self, trial_id: int, state: TrialState) -> None:
        """Set the state of a trial.

        Args:
            trial_id:
                ID of the trial.
            state:
                New state to set.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_value(self, trial_id: int, value: float, state: TrialState) -> None:
        """Set the value of a trial.

        Args:
            trial_id:
                ID of the trial.
            value:
                Value of the trial.
            state:
                New state to set.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_intermediate_value(self, trial_id: int, step: int, value: float) -> None:
        """Set the intermediate value of a trial.

        Args:
            trial_id:
                ID of the trial.
            step:
                Step of the intermediate value.
            value:
                Intermediate value of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Register a user-defined attribute to a trial.

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
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_system_attr(self, trial_id: int, key: str, value: JSONSerializable) -> None:
        """Register an optuna-internal attribute to a trial.

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
        """
        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id: int, deepcopy: bool = True) -> FrozenTrial:
        """Read the data of a trial.

        Args:
            trial_id:
                ID of the trial.
            deepcopy:
                If True, return a deep copy of the trial. Otherwise, return a shallow copy.

        Returns:
            A :class:`~optuna.trial.FrozenTrial` object.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> list[FrozenTrial]:
        """Read a list of trials.

        Args:
            study_id:
                ID of the study.
            deepcopy:
                If True, return a deep copy of the trial. Otherwise, return a shallow copy.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects, sorted by ``trial_id``.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials_by_states(
        self, study_id: int, states: Sequence[TrialState], deepcopy: bool = True
    ) -> list[FrozenTrial]:
        """Read a list of trials by states.

        Args:
            study_id:
                ID of the study.
            states:
                A sequence of states of trials to read.
            deepcopy:
                If True, return a deep copy of the trial. Otherwise, return a shallow copy.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects, sorted by ``trial_id``.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError
