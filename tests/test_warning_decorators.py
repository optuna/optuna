from typing import Any

import pytest

from optuna import warning_decorators
from optuna.warning_decorators import ExperimentalWarning


def _sample_func(_: Any) -> int:

    return 10


class _Sample(object):

    pass


@pytest.mark.parametrize('version', ['1.1.0', '1.1', 100, None])
def test_experimental_decorator(version: str) -> None:

    if version != '1.1.0':
        with pytest.raises(ValueError):
            warning_decorators.experimental(version)
    else:
        decorator_experimental = warning_decorators.experimental(version)
        assert (
            callable(decorator_experimental) and
            decorator_experimental.__name__ == '_experimental'
        )

        decorated_sample_func = decorator_experimental(_sample_func)
        assert decorated_sample_func.__name__ == '_sample_func'
        assert (
            decorated_sample_func.__doc__ ==
            warning_decorators._EXPERIMENTAL_DOCSTRING_TEMPLATE.format(version, version)
        )

        with pytest.warns(ExperimentalWarning):
            decorated_sample_func(None)


@pytest.mark.parametrize('version', ['1.1.0', '1.1', 100, None])
def test_experimental_class_decorator(version: str) -> None:

    if version != '1.1.0':
        with pytest.raises(ValueError):
            warning_decorators.experimental_class(version)
    else:
        decorator_experimental = warning_decorators.experimental_class(version)
        assert (
            callable(decorator_experimental) and
            decorator_experimental.__name__ == '_experimental_class'
        )

        decorated_sample = decorator_experimental(_Sample)
        assert decorated_sample.__name__ == '_Sample'
        assert (
            decorated_sample.__doc__ ==
            warning_decorators._EXPERIMENTAL_DOCSTRING_TEMPLATE.format(version, version)
        )

        with pytest.warns(ExperimentalWarning):
            decorated_sample()
