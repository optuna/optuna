from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from functools import wraps
from inspect import signature
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings


if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _T = TypeVar("_T")


def convert_positional_args(
    *,
    previous_positional_arg_names: Sequence[str],
    warning_stacklevel: int = 2,
) -> "Callable[[Callable[_P, _T]], Callable[_P, _T]]":
    """Convert positional arguments to keyword arguments.

    Args:
        previous_positional_arg_names: List of names previously given as positional arguments.
        warning_stacklevel: Level of the stack trace where decorated function locates.
    """

    def converter_decorator(func: "Callable[_P, _T]") -> "Callable[_P, _T]":
        assert set(previous_positional_arg_names).issubset(set(signature(func).parameters)), (
            f"{set(previous_positional_arg_names)} is not a subset of"
            f" {set(signature(func).parameters)}"
        )

        @wraps(func)
        def converter_wrapper(*args: Any, **kwargs: Any) -> "_T":
            if len(args) >= 1:
                warnings.warn(
                    f"{func.__name__}(): Please give all values as keyword arguments."
                    " See https://github.com/optuna/optuna/issues/3324 for details.",
                    FutureWarning,
                    stacklevel=warning_stacklevel,
                )
            if len(args) > len(previous_positional_arg_names):
                raise TypeError(
                    f"{func.__name__}() takes {len(previous_positional_arg_names)} positional"
                    f" arguments but {len(args)} were given."
                )

            for val, arg_name in zip(args, previous_positional_arg_names):
                # When specifying a positional argument that is not located at the end of args as
                # a keyword argument, raise TypeError as follows by imitating the Python standard
                # behavior.
                if arg_name in kwargs:
                    raise TypeError(
                        f"{func.__name__}() got multiple values for argument '{arg_name}'."
                    )
                kwargs[arg_name] = val

            return func(**kwargs)

        return converter_wrapper

    return converter_decorator
