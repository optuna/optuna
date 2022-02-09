from typing import List

import pytest

from optuna.study.study import _convert_positional_args


def _sample_func(*, a: int, b: int, c: int) -> int:
    return a + b + c


def test_convert_positional_args_decorator() -> None:
    previous_positional_arg_names: List[str] = []
    decorator_converter = _convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )

    decorated_func = decorator_converter(_sample_func)
    assert decorated_func.__name__ == _sample_func.__name__


def test_convert_positional_args_user_warning() -> None:
    previous_positional_arg_names: List[str] = ["a", "b"]
    decorator_converter = _convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    decorated_func = decorator_converter(_sample_func)
    with pytest.warns(FutureWarning) as record:
        decorated_func(1, 2, c=3)  # type: ignore
        decorated_func(1, b=2, c=3)  # type: ignore
        decorated_func(a=1, b=2, c=3)  # no warn

    assert len(record) == 2
    for warn in record.list:
        assert isinstance(warn.message, FutureWarning)
        assert _sample_func.__name__ in str(warn.message)


def test_convert_positional_args_mypy_type_inference() -> None:
    previous_positional_arg_names: List[str] = []
    decorator_converter = _convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    class _Sample(object):
        def __init__(self) -> None:
            pass

        def method(self) -> bool:
            return True

    def _func_sample() -> _Sample:
        return _Sample()

    def _func_none() -> None:
        pass

    ret_none = decorator_converter(_func_none)()
    assert ret_none is None

    ret_sample = decorator_converter(_func_sample)()
    assert isinstance(ret_sample, _Sample)
    assert ret_sample.method()


def test_convert_positional_args_invalid_previous_positional_arg_names() -> None:
    def _test_converter(previous_positional_arg_names: List[str], raise_error: bool) -> None:
        decorator_converter = _convert_positional_args(
            previous_positional_arg_names=previous_positional_arg_names
        )
        assert callable(decorator_converter)

        if raise_error:
            with pytest.raises(AssertionError) as record:
                decorator_converter(_sample_func)
            assert str(record.value) == (
                f"{set(previous_positional_arg_names)} is not a subset of"
                f" {set(['a', 'b', 'c'])}"
            )
        else:
            decorator_converter(_sample_func)

    _test_converter(previous_positional_arg_names=["a", "b", "c", "d"], raise_error=True)
    _test_converter(previous_positional_arg_names=["a", "d"], raise_error=True)
    # Changing the order of the arguments is allowed.
    _test_converter(previous_positional_arg_names=["b", "a"], raise_error=False)


def test_convert_positional_args_invalid_positional_args() -> None:
    previous_positional_arg_names: List[str] = ["a", "b"]
    decorator_converter = _convert_positional_args(
        previous_positional_arg_names=previous_positional_arg_names
    )
    assert callable(decorator_converter)

    decorated_func = decorator_converter(_sample_func)
    with pytest.warns(FutureWarning):
        with pytest.raises(TypeError) as record:
            decorated_func(1, 2, 3)  # type: ignore
        assert str(record.value) == "_sample_func() takes 2 positional arguments but 3 were given."

        with pytest.raises(TypeError) as record:
            decorated_func(1, 3, b=2)  # type: ignore
        assert str(record.value) == "_sample_func() got multiple values for argument 'b'."
