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
    Deprecated in v{ver}. This feature will be removed in the future. See
    https://github.com/optuna/optuna/releases/tag/v{ver}.
"""


def deprecation(version: str, name: str = None) -> Any:
    """Decorate class or function as deprecation.

    Args:
        version: The first version that supports the target feature.
        name: The name of the feature. Defaults to the function or class name. Optional.
    """

    _validate_version(version)

    def _deprecation_wrapper(f: Any) -> Any:
        # f is either func or class.

        def _deprecation_func(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            """Decorates a function as deprecated.

            This decorator is supposed to be applied to the deprecated function.
            """
            if func.__doc__ is None:
                func.__doc__ = ""

            note = _DEPRECATION_NOTE_TEMPLATE.format(ver=version)
            indent = _get_docstring_indent(func.__doc__)
            func.__doc__ = func.__doc__.strip() + textwrap.indent(note, indent) + indent

            # TODO(crcrpar): Annotate this correctly.
            @functools.wraps(func)
            def new_func(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(
                    "{} is deprecated in v{}. "
                    "This feature will be removed in the future.".format(
                        name if name is not None else func.__name__, version
                    ),
                    DeprecationWarning,
                )

                return func(*args, **kwargs)  # type: ignore

            return new_func

        def _deprecation_class(cls: Any) -> Any:
            """Decorates a class as deprecated.

            This decorator is supposed to be applied to the deprecated class.
            """
            _original_init = cls.__init__

            @functools.wraps(_original_init)
            def wrapped_init(self, *args, **kwargs) -> None:  # type: ignore
                warnings.warn(
                    "{} is deprecated in v{}. "
                    "This feature will be removed in the future.".format(
                        name if name is not None else cls.__name__, version
                    ),
                    DeprecationWarning,
                )

                _original_init(self, *args, **kwargs)

            cls.__init__ = wrapped_init

            if cls.__doc__ is None:
                cls.__doc__ = ""

            note = _DEPRECATION_NOTE_TEMPLATE.format(ver=version)
            indent = _get_docstring_indent(cls.__doc__)
            cls.__doc__ = cls.__doc__.strip() + textwrap.indent(note, indent) + indent

            return cls

        return _deprecation_class(f) if inspect.isclass(f) else _deprecation_func(f)

    return _deprecation_wrapper
