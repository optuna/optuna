from __future__ import absolute_import

import colorlog
import logging
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
import threading


_lock = threading.Lock()
_default_handler = None  # type: logging.Handler


def _get_library_name():
    # type: () -> str

    return __name__.split('.')[0]


def _get_library_root_logger():
    # type: () -> logging.Logger

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger():
    # type: () -> None

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured our library root logger.
            return
        _default_handler = colorlog.StreamHandler()
        _default_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s'))

        python_root_logger = logging.getLogger()
        if python_root_logger.handlers:
            # Users have already configured python root logger. Our log outputs will be propagated
            # to the root logger, and thus they will be collected properly. We don't further
            # configure loggers by ourselves to prevent double logging, etc.
            return

        # Apply our default configuration to our library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)


def _reset_library_root_logger():
    # type: () -> None

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name):
    # type: (str) -> logging.Logger

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity():
    # type: () -> int

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity):
    # type: (int) -> None

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler():
    # type: () -> None

    _configure_library_root_logger()
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler():
    # type: () -> None

    _configure_library_root_logger()
    _get_library_root_logger().addHandler(_default_handler)
