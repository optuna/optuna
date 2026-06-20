import logging

import _pytest.capture
import _pytest.logging

import optuna.logging


def test_get_logger(caplog: _pytest.logging.LogCaptureFixture) -> None:
    # Log propagation is necessary for caplog to capture log outputs.
    optuna.logging.enable_propagation()

    logger = optuna.logging.get_logger("optuna.foo")
    with caplog.at_level(logging.INFO, logger="optuna.foo"):
        logger.info("hello")
    assert "hello" in caplog.text


def test_default_handler(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    library_root_logger = optuna.logging._get_library_root_logger()

    example_logger = optuna.logging.get_logger("optuna.bar")

    # Default handler enabled
    optuna.logging.enable_default_handler()
    assert optuna.logging._default_handler in library_root_logger.handlers
    example_logger.info("hey")
    _, err = capsys.readouterr()
    assert "hey" in err

    # Default handler disabled
    optuna.logging.disable_default_handler()
    assert optuna.logging._default_handler not in library_root_logger.handlers
    example_logger.info("yoyo")
    _, err = capsys.readouterr()
    assert "yoyo" not in err


def test_verbosity(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    library_root_logger = optuna.logging._get_library_root_logger()
    example_logger = optuna.logging.get_logger("optuna.hoge")
    optuna.logging.enable_default_handler()

    # level INFO
    optuna.logging.set_verbosity(optuna.logging.INFO)
    assert library_root_logger.getEffectiveLevel() == logging.INFO
    example_logger.warning("hello-warning")
    example_logger.info("hello-info")
    example_logger.debug("hello-debug")
    _, err = capsys.readouterr()
    assert "hello-warning" in err
    assert "hello-info" in err
    assert "hello-debug" not in err

    # level WARNING
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assert library_root_logger.getEffectiveLevel() == logging.WARNING
    example_logger.warning("bye-warning")
    example_logger.info("bye-info")
    example_logger.debug("bye-debug")
    _, err = capsys.readouterr()
    assert "bye-warning" in err
    assert "bye-info" not in err
    assert "bye-debug" not in err


def test_propagation() -> None:
    optuna.logging._reset_library_root_logger()
    logger = optuna.logging.get_logger("optuna.foo")

    # Capture the records that actually reach the root logger to verify propagation directly.
    # We cannot rely on caplog here because, since pytest 9.1.0, caplog captures logs even from
    # non-propagating loggers (https://github.com/pytest-dev/pytest/issues/3697).
    records: list[logging.LogRecord] = []

    class _RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    root_logger = logging.getLogger()
    handler = _RecordingHandler()
    root_logger.addHandler(handler)
    try:
        # Propagation is disabled by default.
        logger.info("no-propagation")
        assert not any(r.getMessage() == "no-propagation" for r in records)

        # Enable propagation.
        optuna.logging.enable_propagation()
        logger.info("enable-propagate")
        assert any(r.getMessage() == "enable-propagate" for r in records)

        # Disable propagation.
        optuna.logging.disable_propagation()
        logger.info("disable-propagation")
        assert not any(r.getMessage() == "disable-propagation" for r in records)
    finally:
        root_logger.removeHandler(handler)
