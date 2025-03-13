from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from functools import wraps
from inspect import Parameter
from inspect import signature
import textwrap
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings

from optuna._deprecated import _validate_two_version
from optuna._experimental import _validate_version


if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _T = TypeVar("_T")

_DEPRECATION_NOTE_TEMPLATE = """

.. warning::
    Using positional arguments is deprecated since v{d_ver}. Support for positional arguments
    will be removed in the future. The removal is currently scheduled for v{r_ver}, but this
    schedule is subject to change. Please use keyword arguments instead.
    See https://github.com/optuna/optuna/releases/tag/v{d_ver}.
"""

_DEPRECATION_WARNING_TEMPLATE = (
    "Positional arguments in {name} have been deprecated since v{d_ver}. "
    "They will be removed in v{r_ver}. "
    "Please use keyword arguments instead. "
    "See https://github.com/optuna/optuna/releases/tag/v{d_ver} for details."
)


def _get_docstring_indent(docstring: str) -> str:
    return docstring.split("\n")[-1] if "\n" in docstring else ""


def _get_positional_arg_names(func: "Callable[_P, _T]") -> list[str]:
    params = signature(func).parameters
    positional_arg_names = [
        name
        for name, p in params.items()
        if p.default == Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD
    ]
    return positional_arg_names


def _infer_kwargs(previous_positional_arg_names: Sequence[str], *args: Any) -> dict[str, Any]:
    inferred_kwargs = {arg_name: val for val, arg_name in zip(args, previous_positional_arg_names)}
    return inferred_kwargs


def convert_positional_args(
    *,
    previous_positional_arg_names: Sequence[str],
    warning_stacklevel: int = 2,
    deprecated_version: str | None = None,
    removed_version: str | None = None,
) -> "Callable[[Callable[_P, _T]], Callable[_P, _T]]":
    """Convert positional arguments to keyword arguments.

    Args:
        previous_positional_arg_names:
            List of names previously given as positional arguments.
        warning_stacklevel:
            Level of the stack trace where decorated function locates.
        deprecated_version:
            Version number in which the use of positional arguments is deprecated.
        removed_version:
            Version number in which the use of positional arguments was completely removed.
    """

    if deprecated_version is None and removed_version is None:
        pass
    else:
        if deprecated_version is None:
            raise ValueError(
                "deprecated_version must not be None when removed_version is specified"
            )
        if removed_version is None:
            raise ValueError(
                "removed_version must not be None when deprecated_version is specified"
            )

        _validate_version(deprecated_version)
        _validate_version(removed_version)
        _validate_two_version(deprecated_version, removed_version)

    def converter_decorator(func: "Callable[_P, _T]") -> "Callable[_P, _T]":
        if deprecated_version or removed_version:
            if func.__doc__ is None:
                func.__doc__ = ""

            note = _DEPRECATION_NOTE_TEMPLATE.format(
                d_ver=deprecated_version, r_ver=removed_version
            )
            indent = _get_docstring_indent(func.__doc__)
            func.__doc__ = func.__doc__.strip() + textwrap.indent(note, indent) + indent

        assert set(previous_positional_arg_names).issubset(set(signature(func).parameters)), (
            f"{set(previous_positional_arg_names)} is not a subset of"
            f" {set(signature(func).parameters)}"
        )

        @wraps(func)
        def converter_wrapper(*args: Any, **kwargs: Any) -> "_T":
            if deprecated_version or removed_version:
                warnings.warn(
                    _DEPRECATION_WARNING_TEMPLATE.format(
                        name=func.__name__,
                        d_ver=deprecated_version,
                        r_ver=removed_version,
                    ),
                    FutureWarning,
                    stacklevel=warning_stacklevel,
                )
            positional_arg_names = _get_positional_arg_names(func)
            inferred_kwargs = _infer_kwargs(previous_positional_arg_names, *args)
            if len(inferred_kwargs) > len(positional_arg_names):
                expected_kwds = set(inferred_kwargs) - set(positional_arg_names)
                warnings.warn(
                    f"{func.__name__}() got {expected_kwds} as positional arguments "
                    "but they were expected to be given as keyword arguments.",
                    FutureWarning,
                    stacklevel=warning_stacklevel,
                )
            if len(args) > len(previous_positional_arg_names):
                raise TypeError(
                    f"{func.__name__}() takes {len(previous_positional_arg_names)} positional"
                    f" arguments but {len(args)} were given."
                )

            duplicated_kwds = set(kwargs).intersection(inferred_kwargs)
            if len(duplicated_kwds):
                # When specifying positional arguments that are not located at the end of args as
                # keyword arguments, raise TypeError as follows by imitating the Python standard
                # behavior
                raise TypeError(
                    f"{func.__name__}() got multiple values for arguments {duplicated_kwds}."
                )

            kwargs.update(inferred_kwargs)

            return func(**kwargs)  # type: ignore[call-arg]

        return converter_wrapper

    return converter_decorator
