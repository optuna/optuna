import pytest

from optuna._imports import try_import


def test_try_import_successful() -> None:
    with try_import() as imports:
        pass

    assert imports.is_successful()
    imports.check()


def test_try_import_failing() -> None:
    # Default arguments.
    with try_import() as imports:
        raise ImportError
    assert not imports.is_successful()
    with pytest.raises(ImportError):
        imports.check()

    # Non-default arguments.
    with try_import(catch=(SyntaxError,)) as imports:
        raise SyntaxError
    assert not imports.is_successful()
    with pytest.raises(SyntaxError):
        imports.check()
