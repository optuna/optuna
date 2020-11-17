from typing import TYPE_CHECKING
import warnings


warnings.warn(
    "`optuna.type_checking` will be removed due to the drop of Python 3.5 support.", FutureWarning
)

__all__ = ["TYPE_CHECKING"]
