from typing import Any
from typing import Callable

from optuna._imports import try_import


with try_import() as _imports:
    import allennlp  # NOQA

# TrainerCallback is conditionally imported because allennlp may be unavailable in
# the environment that builds the documentation.
if _imports.is_successful():
    from optuna.integration._allennlp.dump_best_config import dump_best_config  # NOQA
    from optuna.integration._allennlp.executor import AllenNLPExecutor  # NOQA
    from optuna.integration._allennlp.pruner import AllenNLPPruningCallback  # NOQA
else:
    # I disable mypy here since `allennlp.training.TrainerCallback` is a subclass of `Registrable`
    # (https://docs.allennlp.org/main/api/training/trainer/#trainercallback) but `TrainerCallback`
    # defined here is not `Registrable`, which causes a mypy checking failure.
    class TrainerCallback:  # type: ignore
        """Stub for TrainerCallback."""

        @classmethod
        def register(cls: Any, *args: Any, **kwargs: Any) -> Callable:
            """Stub method for `TrainerCallback.register`.

            This method has the same signature as
            `Registrable.register <https://docs.allennlp.org/master/
            api/common/registrable/#registrable>`_ in AllenNLP.

            """

            def wrapper(subclass: Any, *args: Any, **kwargs: Any) -> Any:
                return subclass

            return wrapper
