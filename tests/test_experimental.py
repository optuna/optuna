from typing import Any

import pytest

from optuna import _experimental
from optuna.exceptions import ExperimentalWarning


def _sample_func(_: Any) -> int:

    return 10


def _f() -> None:
    pass


def _g(a: Any, b: Any = None) -> None:
    pass


def _h(a: Any = None, b: int = 10) -> None:
    pass


class _Sample(object):

    def __init__(self, a: Any, b: Any, c: Any) -> None:
        pass


def test_str() -> None:

    assert _experimental._make_func_spec_str(_f) == '_f()\n\n    '
    assert _experimental._make_func_spec_str(_g) == '_g(a, b=None)\n\n    '
    assert _experimental._make_func_spec_str(_h) == '_h(a=None, b=10)\n\n    '


@pytest.mark.parametrize('version', ['1.1.0', '1.1', 100, None])
def test_experimental_decorator(version: str) -> None:

    if version != '1.1.0':
        with pytest.raises(ValueError):
            _experimental.experimental(version)
    else:
        decorator_experimental = _experimental.experimental(version)
        assert (
            callable(decorator_experimental) and
            decorator_experimental.__name__ == '_experimental_wrapper'
        )

        decorated_sample_func = decorator_experimental(_sample_func)
        assert decorated_sample_func.__name__ == '_sample_func'
        assert (
            decorated_sample_func.__doc__ ==
            _experimental._EXPERIMENTAL_DOCSTRING_TEMPLATE.format(ver=version)
        )

        with pytest.warns(ExperimentalWarning):
            decorated_sample_func(None)


@pytest.mark.parametrize('version', ['1.1.0', '1.1', 100, None])
def test_experimental_class_decorator(version: str) -> None:

    if version != '1.1.0':
        with pytest.raises(ValueError):
            _experimental.experimental(version)
    else:
        decorator_experimental = _experimental.experimental(version)
        assert (
            callable(decorator_experimental) and
            decorator_experimental.__name__ == '_experimental_wrapper'
        )

        decorated_sample = decorator_experimental(_Sample)
        assert decorated_sample.__name__ == '_Sample'
        assert (
            decorated_sample.__doc__ ==
            '__init__(a, b, c)\n\n    ' +
            _experimental._EXPERIMENTAL_DOCSTRING_TEMPLATE.format(ver=version)
        )

        with pytest.warns(ExperimentalWarning):
            decorated_sample('a', 'b', 'c')


def test_experimental_decorator_name() -> None:

    name = 'foo'
    decorator_experimental = _experimental.experimental('1.1.0', name=name)
    decorated_sample = decorator_experimental(_Sample)

    with pytest.warns(ExperimentalWarning) as record:
        decorated_sample('a', 'b', 'c')

    assert name in record.list[0].message.args[0]


def test_experimental_class_decorator_name() -> None:

    name = 'bar'
    decorator_experimental = _experimental.experimental('1.1.0', name=name)
    decorated_sample_func = decorator_experimental(_sample_func)

    with pytest.warns(ExperimentalWarning) as record:
        decorated_sample_func(None)

    assert name in record.list[0].message.args[0]
