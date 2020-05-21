import logging
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
import threading

import colorlog

from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Optional  # NOQA

_lock = threading.Lock()
_default_handler = None  # type: Optional[logging.Handler]


def create_default_formatter() -> colorlog.ColoredFormatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """

    return colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
    )


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.setFormatter(create_default_formatter())

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name.

    This function is not supposed to be directly accessed by library users.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the Optuna's root logger.

    Returns:
        Logging level, e.g., ``optuna.logging.DEBUG`` and ``optuna.logging.INFO``.

    .. note::
        Optuna has following logging levels:

        - ``optuna.logging.CRITICAL``, ``optuna.logging.FATAL``
        - ``optuna.logging.ERROR``
        - ``optuna.logging.WARNING``, ``optuna.logging.WARN``
        - ``optuna.logging.INFO``
        - ``optuna.logging.DEBUG``
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the level for the Optuna's root logger.

    Args:
        verbosity:
            Logging level, e.g., ``optuna.logging.DEBUG`` and ``optuna.logging.INFO``.
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler() -> None:
    """Disable the default handler of the Optuna's root logger.

    Example:

        Stop and then resume logging to :obj:`sys.stderr`.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

        .. testcode::

            import optuna

            study = optuna.create_study()

            # There are no logs in sys.stderr.
            optuna.logging.disable_default_handler()
            study.optimize(objective, n_trials=10)

            # There are logs in sys.stderr.
            optuna.logging.enable_default_handler()
            study.optimize(objective, n_trials=10)
            # [I 2020-02-23 17:00:54,314] Finished trial#10 with value: ...
            # [I 2020-02-23 17:00:54,356] Finished trial#11 with value: ...
            # ...

    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the Optuna's root logger.

    Please refer to the example shown in :func:`~optuna.logging.disable_default_handler()`.
    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.

    Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.

    Please disable the Optuna's default handler to prevent double logging if the root logger has
    been configured.

    Example:

        Propagate all log output to the root logger in order to save them to the file.

        .. testsetup::

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

        .. testcode::

            import optuna
            import logging

            logger = logging.getLogger()

            logger.setLevel(logging.INFO)  # Setup the root logger.
            logger.addHandler(logging.FileHandler("foo.log", mode="w"))

            optuna.logging.enable_propagation()  # Propagate logs to the root logger.
            optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

            study = optuna.create_study()

            logger.info("Start optimization.")
            study.optimize(objective, n_trials=10)

            with open('foo.log') as f:
                assert f.readline() == "Start optimization.\\n"
                assert f.readline().startswith("Finished trial#0 with value:")

    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True
