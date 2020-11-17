from typing import TYPE_CHECKING
import warnings


warnings.warn(
    "`optuna.typing` module will be removed due to drop Python 3.5 support", FutureWarning
)

__all__ = ["TYPE_CHECKING"]
