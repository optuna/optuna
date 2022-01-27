from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import LogUniformDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization import _plotly_imports


__all__ = ["is_available"]


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

            if isinstance(dist, LogUniformDistribution):
                return True

            elif isinstance(dist, (FloatDistribution, IntDistribution)):
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
