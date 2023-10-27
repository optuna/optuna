from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from functools import wraps
from inspect import Parameter
from inspect import signature
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings


if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _T = TypeVar("_T")


def _get_positional_arg_names(func: "Callable[_P, _T]") -> list[str]:
    params = signature(func).parameters
    positional_arg_names = [
        name
        for name, p in params.items()
        if p.default == Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD
    ]
    return positional_arg_names


def _infer_given_args(previous_positional_arg_names: Sequence[str], *args: Any) -> dict[str, Any]:
    inferred_args = {arg_name: val for val, arg_name in zip(args, previous_positional_arg_names)}
    return inferred_args


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
            positional_arg_names = _get_positional_arg_names(func)
            inferred_args = _infer_given_args(previous_positional_arg_names, *args)
            if len(inferred_args) > len(positional_arg_names):
                kwargs_expected = set(inferred_args) - set(positional_arg_names)
                warnings.warn(
                    f"{func.__name__}() got {kwargs_expected} as positional arguments "
                    "but they were expected to be given as keyword arguments."
                    " See https://github.com/optuna/optuna/issues/3324 for details.",
                    FutureWarning,
                    stacklevel=warning_stacklevel,
                )
            if len(args) > len(previous_positional_arg_names):
                raise TypeError(
                    f"{func.__name__}() takes {len(previous_positional_arg_names)} positional"
                    f" arguments but {len(args)} were given."
                )

            duplicated_arg_names = set(kwargs).intersection(inferred_args)
            if len(duplicated_arg_names):
                # When specifying positional arguments that are not located at the end of args as
                # keyword arguments, raise TypeError as follows by imitating the Python standard
                # behavior
                raise TypeError(
                    f"{func.__name__}() got multiple values for arguments {duplicated_arg_names}."
                )

            kwargs.update(inferred_args)

            return func(**kwargs)

        return converter_wrapper

    return converter_decorator
