from packaging import version

from optuna._deprecated import deprecated
from optuna._imports import try_import


with try_import() as _imports:
    import catalyst

    if version.parse(catalyst.__version__) < version.parse("21.3"):
        raise ImportError(
            f"You don't have Catalyst 21.3 installed! Catalyst version: {catalyst.__version__}"
        )
    from catalyst.dl import OptunaPruningCallback

if not _imports.is_successful():
    OptunaPruningCallback = object  # NOQA


@deprecated("2.7.0")
class CatalystPruningCallback(OptunaPruningCallback):  # type: ignore
    """Catalyst callback to prune unpromising trials.


    .. seealso::

        See https://catalyst-team.github.io/catalyst/api/callbacks.html?highlight=optuna#catalyst.callbacks.optuna.OptunaPruningCallback
        for the detailed documentation.

    Args:
        loader_key: loader key for best model selection (based on metric score over the dataset)
        metric_key: metric key for best model selection (based on metric score over the dataset)
        minimize: boolean flag to minimize the required metric
        min_delta: minimal delta for metric improve
        trial:  Optuna.Trial for experiment.
    """  # NOQA

    pass
