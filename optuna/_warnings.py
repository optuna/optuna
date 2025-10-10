from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
from warnings import simplefilter


_OPTUNA_MODULE_ROOT: str = (str(Path(__file__).resolve().parent) + os.sep).casefold()


def find_stacklevel() -> int:
    level = 1
    try:
        while True:
            if (
                not getattr(sys._getframe(level).f_code, "co_filename", "")
                .casefold()
                .startswith(_OPTUNA_MODULE_ROOT)
            ):
                return level
            level += 1
    except ValueError:
        return level


def warn(
    message: str,
    category: type[Warning] = UserWarning,
    stacklevel: int | None = None,
) -> None:
    """
    Warning utility that automatically sets the stacklevel to point to the user code.
    """
    stacklevel = stacklevel or find_stacklevel()
    warnings.warn(message, category, stacklevel=stacklevel)


__all__ = ["warn", "catch_warnings", "filterwarnings", "simplefilter"]
