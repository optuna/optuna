from __future__ import annotations

from functools import wraps
from inspect import Parameter
from inspect import signature
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings

from optuna._deprecated import _validate_two_version
from optuna._experimental import _validate_version


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from typing_extensions import ParamSpec

    _P = ParamSpec("_P")
    _T = TypeVar("_T")


_DEPRECATION_WARNING_TEMPLATE = (
    "Positional arguments {deprecated_positional_arg_names} in {func_name}() "
    "have been deprecated since v{d_ver}. "
    "They will be replaced with the corresponding keyword arguments in v{r_ver}, "
    "so please use the keyword specification instead. "
    "See https://github.com/optuna/optuna/releases/tag/v{d_ver} for details."
)


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
    deprecated_version: str,
    removed_version: str,
    warning_stacklevel: int = 2,
) -> "Callable[[Callable[_P, _T]], Callable[_P, _T]]":
    """Convert positional arguments to keyword arguments.

    Args:
        previous_positional_arg_names:
            List of names previously given as positional arguments.
        warning_stacklevel:
            Level of the stack trace where decorated function locates.
        deprecated_version:
            The version in which the use of positional arguments is deprecated.
        removed_version:
            The version in which the use of positional arguments will be removed.
    """

    if deprecated_version is not None or removed_version is not None:
        if deprecated_version is None:
            raise ValueError(
                "deprecated_version must not be None when removed_version is specified."
            )
        if removed_version is None:
            raise ValueError(
                "removed_version must not be None when deprecated_version is specified."
            )

        _validate_version(deprecated_version)
        _validate_version(removed_version)
        _validate_two_version(deprecated_version, removed_version)

    def converter_decorator(func: "Callable[_P, _T]") -> "Callable[_P, _T]":

        assert set(previous_positional_arg_names).issubset(set(signature(func).parameters)), (
            f"{set(previous_positional_arg_names)} is not a subset of"
            f" {set(signature(func).parameters)}"
        )

        @wraps(func)
        def converter_wrapper(*args: Any, **kwargs: Any) -> "_T":
            warning_messages = []
            positional_arg_names = _get_positional_arg_names(func)
            inferred_kwargs = _infer_kwargs(previous_positional_arg_names, *args)

            if len(inferred_kwargs) > len(positional_arg_names):
                expected_kwds = set(inferred_kwargs) - set(positional_arg_names)
                warning_messages.append(
                    f"{func.__name__}() got {expected_kwds} as positional arguments "
                    "but they were expected to be given as keyword arguments."
                )

                if deprecated_version or removed_version:
                    warning_messages.append(
                        _DEPRECATION_WARNING_TEMPLATE.format(
                            deprecated_positional_arg_names=previous_positional_arg_names,
                            func_name=func.__name__,
                            d_ver=deprecated_version,
                            r_ver=removed_version,
                        )
                    )

            if warning_messages:
                warnings.warn(
                    "\n".join(warning_messages), FutureWarning, stacklevel=warning_stacklevel
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
