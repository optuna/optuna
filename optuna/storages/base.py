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

    Storage class can assume that the returned values are read-only. Additionally, storage
    classes can assume that a single process has at most one storage. However, the storage
    might be shared from multiple threads and storage classes must be thread-safe.

    Storage classes must support monotonic-reads consistency model, that is, if a
    process reads a data `X`, any successive reads on data `X` does not return
    older values.
    They must virtually support read-your-writes, that is, if a process writes to
    data `X`, any successive reads on data `X` from the same process must read
    the written value or one of more recent values.
    If `sync` method is called, any successive reads on `state` attribute
    of `Trial` and any attributes of `Study` must be equal or more recent than the value
    at the time that the `sync` method is called.
    If `sync` method is called, any successive reads on any attributes of `Trial` `T`
    must return the same or more recent value than any preceding writes to `T` via
    `update_trial` method.
    Writes on any attributes of `Trial` must be completed before any
    successive writes on the `state` attribute of the `Trial` with the same trial ID.

    Under assumptions that
    1. there are no process failures,
    2. every trial eventually terminates, and
    3. each trial is updated from only one process,
    storage classes must support eventual-consistency, monotonic-reads,
    monotonic-writes, and read-your-writes consistency models.

    Raises:
        KeyError:
            On reads and writes on non-existing studies and trials,
            storage classes always raise KeyError.
    """

    # Basic study manipulation

    @abc.abstractmethod
    def create_new_study(self, study_name: Optional[str] = None) -> int:

        """Creates a new study with a given name.

        When no name is specified, storage class auto-generates the name.
        Study ID is unique among all current and deleted studies.

        Args:
            study_name:
                Name of a new study to create.

        Returns:
            Study ID, which is an unique id among studies, of the created study.

        Raises (current implementation):
            rdb:
                optuna.exceptions.DuplicatedStudyError:
                    If a study with the same `study_name` already exists.
            redis:
                optuna.exceptions.DuplicatedStudyError:
                    If a study with the same `study_name` already exists.

        Raises (proposal):
            optuna.exceptions.DuplicatedStudyError:
                If a study with the same `study_name` already exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def delete_study(self, study_id: int) -> None:

        """Deletes a study specified by the study ID.

        Trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                Study ID of a study to delete.

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                ValueError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            in-memory:
                ValueError
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError
                    If no study with the matching `study_id` exists.
            redis:
                ValueError
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_direction(self, study_id: int, direction: study.StudyDirection) -> None:

        """Register a direction of optimization problem to a study.

        Args:
            study_id:
                Study ID of a study to update.
            direction:
                Either StudyDirection.MAXIMIZE or StudyDirection.MINIMIZE.

        Raises:
            in-memory:
                ValueError
                    If no study with the matching `study_id` exists.
                    If `direction` is already set and the passed `direction` conflicts with it.
            rdb:
                ValueError
                    If no study with the matching `study_id` exists.
                    If `direction` is already set and the passed `direction` conflicts with it.
            redis:
                ValueError
                    If no study with the matching `study_id` exists.
                    If `direction` is already set and the passed `direction` conflicts with it.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
            ValueError:
                If `direction` is already set and the passed `direction` conflicts with it.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        """Register an ontuna-internal attribute to a study.

        This method overwrites an existing attribute.

        Args:
            study_id:
                Study ID of a study to update.
            key:
                Key of an attribute.
            value:
                Attributes' value.

        Raises (current implementation):
            in-memory:
                ValueError
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError
                    If no study with the matching `study_id` exists.
            redis:
                ValueError
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                ValueError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
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

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                AssertionError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                AssertionError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            redis:
                AssertionError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
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

        Raises (current implementation):
            redis:
                AssertionError

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_study_summaries(self) -> List[study.StudySummary]:

        """Returns a list of `study.StudySummary` objects.

        Returns:
            A list of `study.StudySummary` objects.

        """

        raise NotImplementedError

    # Basic trial manipulation

    @abc.abstractmethod
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial]=None) -> int:

        """Create and add a new trial to a specified study.

        Args:
            study_id:
                Study ID to add a new trial.
            template_trial:
                Fronzen trial with default user-attributes, system-attributes,
                intermediate-values, and a state.

        Returns:
            Trial ID of the created trial.

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                sqlalchemy.exc.IntegrityError:
                    If no study with the matching `study_id` exists.
            redis:
                AssertionError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:

        """Update a state of a specified trial.

        Args:
            trial_id:
                Trial ID of a trial to update its state.
            state:
                New state of the trial.

        Returns:

        Raise (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
                If the trial is already finished.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_param(self, trial_id: int, param_name: str, param_value_internal: float, distribution: distributions.BaseDistribution) -> bool:

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
            Return False when the parameter is already set to the trial.

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
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

        Raises (current implementation):
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            in-memory:
                no check (lambda x: x)
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

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                KeyError:
                    If no such parameter exists.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                    If no such parameter exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                KeyError:
                    If no such parameter exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
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

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
                If the trial is already finished.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def set_trial_intermediate_value(self, trial_id: int, step: int, intermediate_value: float) -> bool:

        """Report a value within an evaluation of an objective function.

        Args:
            trial_id:
                Trial ID of a trial to set intermediate value.
            step:
                Step of the trial (e.g., Epoch of neural network training).
            intermediate_value:
                Reported value within an evaluation of an objective function.

        Returns:
            Return False when the intermediate of the step already exists.

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
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

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
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

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.
                RuntimeError:
                    If the trial is already finished.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
            RuntimeError:
                If the trial is already finished.
        """

        raise NotImplementedError

    # Basic trial access

    @abc.abstractmethod
    def get_trial(self, trial_id: int) -> FrozenTrial:

        """Read a trial using a trial ID.

        Args:
            trial_id:
                Trial ID of a trial to read.

        Returns:
            Trial with a matching trial ID.

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_all_trials(self, study_id: int, deepcopy: bool=True) -> List[FrozenTrial]:

        """Read all trials in a specified study.

        Args:
            study_id:
                Study ID of a study to read trials from.
            deepcopy:
                Whether copy the list of trials before returning.
                Set True when you might update the list or elements of the list.

        Returns:
            A list of trials in the study.

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                ValueError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_n_trials(self, study_id: int, state: Optional[TrialState]=None) -> int:

        """Count the number of trials in a specified study.

        Args:
            study_id:
                Study ID of a study to count trials.
            state:

        Returns:
            Number of the trials in the study.

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no study with the matching `study_id` exists.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
            redis:
                ValueError:
                    If no study with the matching `study_id` exists.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
        """

        raise NotImplementedError

    def get_best_trial(self, study_id :int) -> FrozenTrial:

        """Return a trial with the best value in the study.

        Args:
            study_id:
                Study ID to search the best trial.

        Returns:
            The trial with the best return value of the objective function
            among all finished tirals in the study.

        Raises (current implementation):
            in-memory:
                ValueError:
                    If no trials have been completed.
            rdb:
                ValueError:
                    If no study with the matching `study_id` exists.
                    If no trials have been completed.
            redis:
                ValueError:
                    If no study with the matching `study_id` exists.
                    If no trials have been completed.

        Raises (proposal):
            KeyError:
                If no study with the matching `study_id` exists.
            RuntimeError:
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

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
        """

        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id: int) -> Dict[str, Any]:

        """Read a user-defined attributes of a specified trial.

        Args:
            trial_id:
                A trial ID of a trial to read user-defined attributes.

        Returns:
            A dictionary of user-defined attributes of the trial.

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
        """

        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id: int) -> Dict[str, Any]:

        """Read an optuna-internal attributes of a specified trial.

        Args:
            trial_id:
                A trial ID of a trial to read optuna-internal attributes.

        Returns:
            A dictionary of optuna-internal attributes of the trial.

        Raises (current implementation):
            in-memory:
                IndexError:
                    If no trial with the matching `trial_id` exists.
            rdb:
                ValueError:
                    If no trial with the matching `trial_id` exists.
            redis:
                AssertionError:
                    If no trial with the matching `trial_id` exists.

        Raises (proposal):
            KeyError:
                If no trial with the matching `trial_id` exists.
        """

        return self.get_trial(trial_id).system_attrs

    def sync(self) -> None:

        """Load data from remote database if exists."""

        pass

    def update_trial(
        self,
        trial_id: int,
        state: Optional[TrialState] = None,
        value: Optional[float] = None,
        values: Optional[Dict[int, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        dists: Optional[Dict[str, distributions.BaseDistribution]] = None,
        user_attrs: Optional[Dict[str, Any]] = None,
        system_attrs: Optional[Dict[str, Any]] = None,
    ) -> bool:

        """Sync latest trial updates to a database.

        Args:
            trial_id:
                Trial id of the trial to update.
            state:
                New state. None when there are no changes.
            value:
                New value. None when there are no changes.
            values:
                New intermediate values. None when there are no updates.
            params:
                New parameter dictionary. None when there are no updates.
            dists:
                New parameter distributions. None when there are no updates.
            user_attrs:
                New user_attr. None when there are no updates.
            system_attrs:
                New system_attr. None when there are no updates.

        Returns:
            True when success.

        """

        raise NotImplementedError

    def remove_session(self) -> None:

        """Clean up all connections to a database."""

        pass

    def check_trial_is_updatable(self, trial_id: int, trial_state: TrialState) -> None:

        """Check whether a `trial_state` is updatable.

        Args:
            trial_id:
                Trial id of a trial to update.
                Only used for an error message.
            trial_state:
                Trial state to check.

        Raises:

        """

        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number)
            )
