import sys

import pytest


pytest.register_assert_rewrite("optuna.testing.pytest_samplers")
pytest.register_assert_rewrite("optuna.testing.pytest_storages")


def pytest_runtest_setup() -> None:
    if sys.platform != "win32":
        return

    import optuna.logging

    optuna.logging._reset_library_root_logger()
    optuna.logging._configure_library_root_logger()
