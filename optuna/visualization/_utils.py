from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union
import warnings

import numpy as np

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.visualization import _plotly_imports


__all__ = ["is_available"]
_logger = optuna.logging.get_logger(__name__)


def is_available() -> bool:
    """Returns whether visualization with plotly is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on plotly version 4.0.0 or higher. If a
        supported version of plotly isn't installed in your environment, this function will return
        :obj:`False`. In such case, please execute ``$ pip install -U plotly>=4.0.0`` to install
        plotly.

    Returns:
        :obj:`True` if visualization with plotly is available, :obj:`False` otherwise.
    """

    return _plotly_imports._imports.is_successful()


if is_available():
    import plotly.colors

    COLOR_SCALE = plotly.colors.sequential.Blues


def _check_plot_args(
    study: Union[Study, Sequence[Study]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> None:

    studies: Sequence[Study]
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = study

    if target is None and any(study._is_multi_objective() for study in studies):
        raise ValueError(
            "If the `study` is being used for multi-objective optimization, "
            "please specify the `target`."
        )

    if target is not None and target_name == "Objective Value":
        warnings.warn(
            "`target` is specified, but `target_name` is the default value, 'Objective Value'."
        )


def _is_log_scale(trials: List[FrozenTrial], param: str) -> bool:

    for trial in trials:
        if param in trial.params:
            dist = trial.distributions[param]

            if isinstance(dist, (FloatDistribution, IntDistribution)):
                if dist.log:
                    return True

    return False


def _is_categorical(trials: List[FrozenTrial], param: str) -> bool:

    return any(
        isinstance(t.distributions[param], CategoricalDistribution)
        for t in trials
        if param in t.params
    )


def _is_numerical(trials: List[FrozenTrial], param: str) -> bool:
    return all(
        (isinstance(t.params[param], int) or isinstance(t.params[param], float))
        and not isinstance(t.params[param], bool)
        for t in trials
        if param in t.params
    )


def _get_param_values(trials: List[FrozenTrial], p_name: str) -> List[Any]:

    values = [t.params[p_name] for t in trials if p_name in t.params]
    if _is_numerical(trials, p_name):
        return values
    return list(map(str, values))


def _get_skipped_trial_numbers(
    trials: List[FrozenTrial], used_param_names: Sequence[str]
) -> Set[int]:
    """Utility function for ``plot_parallel_coordinate``.

    If trial's parameters do not contain a parameter in ``used_param_names``,
    ``plot_parallel_coordinate`` methods do not use such trials.

    Args:
        trials:
            List of ``FrozenTrial``s.
        used_param_names:
            The parameter names used in ``plot_parallel_coordinate``.

    Returns:
        A set of invalid trial numbers.
    """

    skipped_trial_numbers = set()
    for trial in trials:
        for used_param in used_param_names:
            if used_param not in trial.params.keys():
                skipped_trial_numbers.add(trial.number)
                break
    return skipped_trial_numbers


def _filter_nonfinite(
    trials: List[FrozenTrial],
    target: Optional[Callable[[FrozenTrial], float]] = None,
    with_message: bool = True,
) -> List[FrozenTrial]:

    # For multi-objective optimization target must be specified to select
    # one of objective values to filter trials by (and plot by later on).
    # This function is not raising when target is missing, since we're
    # assuming plot args have been sanitized before.
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    filtered_trials: List[FrozenTrial] = []
    for trial in trials:
        value = target(trial)

        try:
            value = float(value)
        except (
            ValueError,
            TypeError,
        ):
            warnings.warn(
                f"Trial{trial.number}'s target value {repr(value)} could not be cast to float."
            )
            raise

        # Not a Number, positive infinity and negative infinity are considered to be non-finite.
        if not np.isfinite(value):
            if with_message:
                _logger.warning(
                    f"Trial {trial.number} is omitted in visualization "
                    "because its objective value is inf or nan."
                )
        else:
            filtered_trials.append(trial)

    return filtered_trials


def _is_reverse_scale(study: Study, target: Optional[Callable[[FrozenTrial], float]]) -> bool:

    return target is not None or study.direction == StudyDirection.MINIMIZE
