from packaging import version

from optuna._deprecated import deprecated_class
from optuna._imports import try_import


with try_import() as _imports:
    import catalyst

    if version.parse(catalyst.__version__) < version.parse("21.3"):
        raise ImportError(
            f"You don't have Catalyst>=21.3 installed! Catalyst version: {catalyst.__version__}"
        )
    from catalyst.dl import OptunaPruningCallback

if not _imports.is_successful():
    OptunaPruningCallback = object  # NOQA


@deprecated_class("2.7.0", "4.0.0")
class CatalystPruningCallback(OptunaPruningCallback):
    """Catalyst callback to prune unpromising trials.

    This class is an alias to Catalyst's
    `OptunaPruningCallback <https://catalyst-team.github.io/catalyst/api/callbacks.html?highlight=optuna#catalyst.callbacks.optuna.OptunaPruningCallback>`_.

    See the Catalyst's documentation for the detailed description.
    """  # NOQA

    pass
