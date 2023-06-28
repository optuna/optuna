import copy
import math
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import optuna
from optuna import logging
from optuna import pruners
from optuna import trial as trial_module
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# This is used for propagating warning message to Study.optimize.
STUDY_TELL_WARNING_KEY = "STUDY_TELL_WARNING"


_logger = logging.get_logger(__name__)


def _get_frozen_trial(study: "optuna.Study", trial: Union[trial_module.Trial, int]) -> FrozenTrial:
    if isinstance(trial, trial_module.Trial):
        trial_id = trial._trial_id
    elif isinstance(trial, int):
        trial_number = trial
        try:
            trial_id = study._storage.get_trial_id_from_study_id_trial_number(
                study._study_id, trial_number
            )
        except KeyError as e:
            raise ValueError(
                f"Cannot tell for trial with number {trial_number} since it has not been "
                "created."
            ) from e
    else:
        raise TypeError("Trial must be a trial object or trial number.")

    return study._storage.get_trial(trial_id)


def _check_state_and_values(
    state: Optional[TrialState], values: Optional[Union[float, Sequence[float]]]
) -> None:
    if state == TrialState.COMPLETE:
        if values is None:
            raise ValueError(
                "No values were told. Values are required when state is TrialState.COMPLETE."
            )
    elif state in (TrialState.PRUNED, TrialState.FAIL):
        if values is not None:
            raise ValueError(
                "Values were told. Values cannot be specified when state is "
                "TrialState.PRUNED or TrialState.FAIL."
            )
    elif state is not None:
        raise ValueError(f"Cannot tell with state {state}.")


def _check_values_are_feasible(study: "optuna.Study", values: Sequence[float]) -> Optional[str]:
    for v in values:
        # TODO(Imamura): Construct error message taking into account all values and do not early
        # return `value` is assumed to be ignored on failure so we can set it to any value.
        try:
            float(v)
        except (ValueError, TypeError):
            return f"The value {repr(v)} could not be cast to float"

        if math.isnan(v):
            return f"The value {v} is not acceptable"

    if len(study.directions) != len(values):
        return (
            f"The number of the values {len(values)} did not match the number of the objectives "
            f"{len(study.directions)}"
        )

    return None


def _tell_with_warning(
    study: "optuna.Study",
    trial: Union[trial_module.Trial, int],
    value_or_values: Optional[Union[float, Sequence[float]]] = None,
    state: Optional[TrialState] = None,
    skip_if_finished: bool = False,
    suppress_warning: bool = False,
) -> FrozenTrial:
    """Internal method of :func:`~optuna.study.Study.tell`.

    Refer to the document for :func:`~optuna.study.Study.tell` for the reference.
    This method has one additional parameter ``suppress_warning``.

    Args:
        suppress_warning:
            If :obj:`True`, tell will not show warnings when tell receives an invalid
            values. This flag is expected to be :obj:`True` only when it is invoked by
            Study.optimize.
    """

    # We must invalidate all trials cache here as it is only valid within a trial.
    study._thread_local.cached_all_trials = None

    # Validate the trial argument.
    frozen_trial = _get_frozen_trial(study, trial)
    if frozen_trial.state.is_finished() and skip_if_finished:
        _logger.info(
            f"Skipped telling trial {frozen_trial.number} with values "
            f"{value_or_values} and state {state} since trial was already finished. "
            f"Finished trial has values {frozen_trial.values} and state {frozen_trial.state}."
        )
        return copy.deepcopy(frozen_trial)
    elif frozen_trial.state != TrialState.RUNNING:
        raise ValueError(f"Cannot tell a {frozen_trial.state.name} trial.")

    # Validate the state and values arguments.
    values: Optional[Sequence[float]]
    if value_or_values is None:
        values = None
    elif isinstance(value_or_values, Sequence):
        values = value_or_values
    else:
        values = [value_or_values]

    _check_state_and_values(state, values)

    warning_message = None

    if state == TrialState.COMPLETE:
        assert values is not None

        values_conversion_failure_message = _check_values_are_feasible(study, values)
        if values_conversion_failure_message is not None:
            raise ValueError(values_conversion_failure_message)
    elif state == TrialState.PRUNED:
        # Register the last intermediate value if present as the value of the trial.
        # TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
        assert values is None

        last_step = frozen_trial.last_step
        if last_step is not None:
            last_intermediate_value = frozen_trial.intermediate_values[last_step]
            # intermediate_values can be unacceptable value, i.e., NaN.
            if _check_values_are_feasible(study, [last_intermediate_value]) is None:
                values = [last_intermediate_value]
    elif state is None:
        if values is None:
            values_conversion_failure_message = "The value None could not be cast to float."
        else:
            values_conversion_failure_message = _check_values_are_feasible(study, values)

        if values_conversion_failure_message is None:
            state = TrialState.COMPLETE
        else:
            state = TrialState.FAIL
            values = None
            if not suppress_warning:
                warnings.warn(values_conversion_failure_message)
            else:
                warning_message = values_conversion_failure_message

    assert state is not None

    # Cast values to list of floats.
    if values is not None:
        # values have beed checked to be castable to floats in _check_values_are_feasible.
        values = [float(value) for value in values]

    # Post-processing and storing the trial.
    try:
        # Sampler defined trial post-processing.
        study = pruners._filter_study(study, frozen_trial)
        study.sampler.after_trial(study, frozen_trial, state, values)
    finally:
        study._storage.set_trial_state_values(frozen_trial._trial_id, state, values)

    frozen_trial = copy.deepcopy(study._storage.get_trial(frozen_trial._trial_id))

    if warning_message is not None:
        frozen_trial._system_attrs[STUDY_TELL_WARNING_KEY] = warning_message
    return frozen_trial
