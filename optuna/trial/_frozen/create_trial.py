import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

from optuna import logging
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.trial._frozen.base import BaseFrozenTrial
from optuna.trial._frozen.multi import MultiObjectiveFrozenTrial
from optuna.trial._frozen.single import FrozenTrial
from optuna.trial._state import TrialState


_logger = logging.get_logger(__name__)


@experimental("2.0.0")
def create_trial(
    *,
    state: Optional[TrialState] = None,
    value: Optional[Union[float, Sequence[float]]] = None,
    params: Optional[Dict[str, Any]] = None,
    distributions: Optional[Dict[str, BaseDistribution]] = None,
    user_attrs: Optional[Dict[str, Any]] = None,
    system_attrs: Optional[Dict[str, Any]] = None,
    intermediate_values: Optional[Dict[int, Union[float, Sequence[float]]]] = None,
) -> BaseFrozenTrial:
    """Create a new :class:`~optuna.trial.FrozenTrial` or :class:`~optuna.trial.MultiObjectiveFrozenTrial`.

    Example:
        .. testcode::
            import optuna
            from optuna.distributions import CategoricalDistribution
            from optuna.distributions import UniformDistribution
            trial = optuna.trial.create_trial(
                params={"x": 1.0, "y": 0},
                distributions={
                    "x": UniformDistribution(0, 10),
                    "y": CategoricalDistribution([-1, 0, 1]),
                },
                value=5.0,
            )
            assert isinstance(trial, optuna.trial.FrozenTrial)
            assert trial.value == 5.0
            assert trial.params == {"x": 1.0, "y": 0}
    .. seealso::
        See :func:`~optuna.study.Study.add_trial` for how this function can be used to create a
        study from existing trials.
    .. note::
        Please note that this is a low-level API. In general, trials that are passed to objective
        functions are created inside :func:`~optuna.study.Study.optimize`.

    .. note::
        If ``value`` and ``intermediate_values`` are both not specified, the
        :class:`~optuna.trial.FrozenTrial` is returned.

    Args:
        state:
            Trial state.
        value:
            Trial objective value. Must be specified if ``state`` is :class:`TrialState.COMPLETE`.
        params:
            Dictionary with suggested parameters of the trial.
        distributions:
            Dictionary with parameter distributions of the trial.
        user_attrs:
            Dictionary with user attributes.
        system_attrs:
            Dictionary with system attributes. Should not have to be used for most users.
        intermediate_values:
            Dictionary with intermediate objective values of the trial.
    Returns:
        Created trial.
    """

    params = params or {}
    distributions = distributions or {}
    user_attrs = user_attrs or {}
    system_attrs = system_attrs or {}
    intermediate_values = intermediate_values or {}
    state = state or TrialState.COMPLETE

    datetime_start = datetime.datetime.now()
    if state.is_finished():
        datetime_complete: Optional[datetime.datetime] = datetime_start
    else:
        datetime_complete = None

    if _is_multi_objective(value, intermediate_values):
        trial = MultiObjectiveFrozenTrial(
            number=-1,
            trial_id=-1,
            state=state,
            value=value,
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
            intermediate_values=intermediate_values,
        )
    else:
        trial = FrozenTrial(
            number=-1,
            trial_id=-1,
            state=state,
            value=value,
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
            intermediate_values=intermediate_values,
        )

    trial._validate()

    return trial


@experimental("2.4.0")
def create_multi_objective_trial(
    *,
    state: Optional[TrialState] = None,
    value: Optional[Sequence[float]] = None,
    params: Optional[Dict[str, Any]] = None,
    distributions: Optional[Dict[str, BaseDistribution]] = None,
    user_attrs: Optional[Dict[str, Any]] = None,
    system_attrs: Optional[Dict[str, Any]] = None,
    intermediate_values: Optional[Dict[int, Sequence[float]]] = None,
) -> BaseFrozenTrial:
    """Create a new :class:`~optuna.trial.MultiObjectiveFrozenTrial`.

    Example:
        .. testcode::
            import optuna
            from optuna.distributions import CategoricalDistribution
            from optuna.distributions import UniformDistribution
            trial = optuna.trial.create_multi_objective_trial(
                params={"x": 1.0, "y": 0},
                distributions={
                    "x": UniformDistribution(0, 10),
                    "y": CategoricalDistribution([-1, 0, 1]),
                },
                value=(5.0, 10.0),
            )
            assert isinstance(trial, optuna.trial.FrozenTrial)
            assert trial.value == (5.0, 10.0)
            assert trial.params == {"x": 1.0, "y": 0}

    .. seealso::
        See :func:`~optuna.study.Study.add_trial` for how this function can be used to create a
        study from existing trials.

    .. note::
        Please note that this is a low-level API. In general, trials that are passed to objective
        functions are created inside :func:`~optuna.study.Study.optimize`.

    Args:
        state:
            Trial state.
        value:
            Trial objective value. Must be specified if ``state`` is :class:`TrialState.COMPLETE`.
        params:
            Dictionary with suggested parameters of the trial.
        distributions:
            Dictionary with parameter distributions of the trial.
        user_attrs:
            Dictionary with user attributes.
        system_attrs:
            Dictionary with system attributes. Should not have to be used for most users.
        intermediate_values:
            Dictionary with intermediate objective values of the trial.
    Returns:
        Created trial.
    """

    params = params or {}
    distributions = distributions or {}
    user_attrs = user_attrs or {}
    system_attrs = system_attrs or {}
    intermediate_values = intermediate_values or {}
    state = state or TrialState.COMPLETE

    datetime_start = datetime.datetime.now()
    if state.is_finished():
        datetime_complete: Optional[datetime.datetime] = datetime_start
    else:
        datetime_complete = None

    trial = MultiObjectiveFrozenTrial(
        number=-1,
        trial_id=-1,
        state=state,
        value=value,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
    )

    trial._validate()

    return trial


def _is_multi_objective(
    value: Optional[Union[float, Sequence[float]]],
    intermediate_values: Optional[Dict[int, Union[float, Sequence[float]]]],
) -> bool:
    if value is not None and isinstance(value, Sequence):
        return True
    if intermediate_values is not None and any(
        [isinstance(v, Sequence) for v in intermediate_values.values()]
    ):
        return True
    if value is None and intermediate_values is None:
        _logger.warning(
            "The ``value`` and ``intermediate_values`` are both ``None``. It is impossible to "
            "determine whether the problem is single-objective or multi-objective. `FrozenTrial`"
            " is returned. If you want `MultiObjectiveFrozenTrial`, please use "
            "`create_multi_objective_trial` instead."
        )
        return False
    return False
