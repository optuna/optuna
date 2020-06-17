import functools
import inspect
import textwrap
from typing import Any
from typing import Callable
import warnings

from optuna._experimental import _get_docstring_indent
from optuna._experimental import _validate_version


_DEPRECATION_NOTE_TEMPLATE = """

.. note::
    Deprecated in v{d_ver}. This feature will be removed in the future. The removal of this
    feature is currently scheduled for v{r_ver}, but this schedule is subject to change.
    See https://github.com/optuna/optuna/releases/tag/v{d_ver}.
"""


def deprecated(deprecated_version: str, removed_version: str, name: str = None) -> Any:
    """Decorate class or function as deprecated.

    Args:
        deprecated_version: The version in which the target feature is deprecated.
        removed_version: The version in which the target feature will be removed.
        name: The name of the feature. Defaults to the function or class name. Optional.
    """

    _validate_version(deprecated_version)
    _validate_version(removed_version)

    def _deprecated_wrapper(f: Any) -> Any:
        # f is either func or class.

        def _deprecated_func(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            """Decorates a function as deprecated.

            This decorator is supposed to be applied to the deprecated function.
            """
            if func.__doc__ is None:
                func.__doc__ = ""

            note = _DEPRECATION_NOTE_TEMPLATE.format(
                d_ver=deprecated_version, r_ver=removed_version
            )
            indent = _get_docstring_indent(func.__doc__)
            func.__doc__ = func.__doc__.strip() + textwrap.indent(note, indent) + indent

            # TODO(mamu): Annotate this correctly.
            @functools.wraps(func)
            def new_func(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(
                    "{} has been deprecated in v{}. "
                    "This feature will be removed in v{}.".format(
                        name if name is not None else func.__name__,
                        deprecated_version,
                        removed_version,
                    ),
                    DeprecationWarning,
                )

                return func(*args, **kwargs)  # type: ignore

            return new_func

        def _deprecated_class(cls: Any) -> Any:
            """Decorates a class as deprecated.

            This decorator is supposed to be applied to the deprecated class.
            """
            _original_init = cls.__init__

            @functools.wraps(_original_init)
            def wrapped_init(self, *args, **kwargs) -> None:  # type: ignore
                warnings.warn(
                    "{} has been deprecated in v{}. "
                    "This feature will be removed in v{}.".format(
                        name if name is not None else cls.__name__,
                        deprecated_version,
                        removed_version,
                    ),
                    DeprecationWarning,
                )

                _original_init(self, *args, **kwargs)

            cls.__init__ = wrapped_init

            if cls.__doc__ is None:
                cls.__doc__ = ""

            note = _DEPRECATION_NOTE_TEMPLATE.format(
                d_ver=deprecated_version, r_ver=removed_version
            )
            indent = _get_docstring_indent(cls.__doc__)
            cls.__doc__ = cls.__doc__.strip() + textwrap.indent(note, indent) + indent

            return cls

        return _deprecated_class(f) if inspect.isclass(f) else _deprecated_func(f)

    return _deprecated_wrapper
