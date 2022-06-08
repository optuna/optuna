import functools
import textwrap
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
import warnings

from typing_extensions import ParamSpec

from optuna.exceptions import ExperimentalWarning


FT = TypeVar("FT")
FP = ParamSpec("FP")
CT = TypeVar("CT")

_EXPERIMENTAL_NOTE_TEMPLATE = """

.. note::
    Added in v{ver} as an experimental feature. The interface may change in newer versions
    without prior notice. See https://github.com/optuna/optuna/releases/tag/v{ver}.
"""


def _validate_version(version: str) -> None:

    if not isinstance(version, str) or len(version.split(".")) != 3:
        raise ValueError(
            "Invalid version specification. Must follow `x.y.z` format but `{}` is given".format(
                version
            )
        )


def _get_docstring_indent(docstring: str) -> str:
    return docstring.split("\n")[-1] if "\n" in docstring else ""


def experimental_func(
    version: str,
    name: Optional[str] = None,
) -> Callable[[Callable[FP, FT]], Callable[FP, FT]]:
    """Decorate function as experimental.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to the function name. Optional.
    """

    _validate_version(version)

    def decorator(func: Callable[FP, FT]) -> Callable[FP, FT]:
        if func.__doc__ is None:
            func.__doc__ = ""

        note = _EXPERIMENTAL_NOTE_TEMPLATE.format(ver=version)
        indent = _get_docstring_indent(func.__doc__)
        func.__doc__ = func.__doc__.strip() + textwrap.indent(note, indent) + indent

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> FT:
            warnings.warn(
                "{} is experimental (supported from v{}). "
                "The interface can change in the future.".format(
                    name if name is not None else func.__name__, version
                ),
                ExperimentalWarning,
                stacklevel=2,
            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def experimental_class(
    version: str,
    name: Optional[str] = None,
) -> Callable[[CT], CT]:
    """Decorate class as experimental.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to the class name. Optional.
    """

    _validate_version(version)

    def decorator(cls: CT) -> CT:
        def wrapper(cls: CT) -> CT:
            """Decorates a class as experimental.

            This decorator is supposed to be applied to the experimental class.
            """
            _original_init = getattr(cls, "__init__")
            _original_name = getattr(cls, "__name__")

            @functools.wraps(_original_init)
            def wrapped_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
                warnings.warn(
                    "{} is experimental (supported from v{}). "
                    "The interface can change in the future.".format(
                        name if name is not None else _original_name, version
                    ),
                    ExperimentalWarning,
                    stacklevel=2,
                )

                _original_init(self, *args, **kwargs)

            setattr(cls, "__init__", wrapped_init)

            if cls.__doc__ is None:
                cls.__doc__ = ""

            note = _EXPERIMENTAL_NOTE_TEMPLATE.format(ver=version)
            indent = _get_docstring_indent(cls.__doc__)
            cls.__doc__ = cls.__doc__.strip() + textwrap.indent(note, indent) + indent

            return cls

        return wrapper(cls)

    return decorator
