from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings


_OPTUNA_MODULE_ROOT: str = str(Path(__file__).resolve().parent) + os.sep


def optuna_warn(
    message: str,
    category: type[Warning] = UserWarning,
    stacklevel: int = 1,
) -> None:
    """
    Wrapper for :func:`warnings.warn` that hides internal Optuna stack frames (for Python 3.12+).

    Behavior:
        - Python 3.12+:
            Uses `skip_file_prefixes` so that warnings appear to originate
            from the user's calling code rather than inside Optuna.
        - Python <3.12:
            This function behaves exactly the same as calling `warnings.warn`
            directly, with no stack frame suppression.
    """

    if sys.version_info >= (3, 12):
        warnings.warn(
            message,
            category=category,
            stacklevel=stacklevel,
            skip_file_prefixes=(_OPTUNA_MODULE_ROOT,),
        )
    else:
        # Increase stacklevel by 1 to account for this wrapper function.
        warnings.warn(message, category, stacklevel=stacklevel + 1)


__all__ = ["optuna_warn"]
