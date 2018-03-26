import _pytest.capture  # NOQA
import _pytest.logging  # NOQA
import logging

import pfnopt.logging


def test_get_logger(caplog):
    # type: (_pytest.logging.LogCaptureFixture) -> None

    logger = pfnopt.logging.get_logger('pfnopt.foo')
    with caplog.at_level(logging.INFO, logger='pfnopt.foo'):
        logger.info('hello')
    assert 'hello' in caplog.text


def test_default_handler(capsys):
    # type: (_pytest.capture.CaptureFixture) -> None

    # We need to reconstruct our default handler to properly capture stderr.
    pfnopt.logging._reset_library_root_logger()
    library_root_logger = pfnopt.logging._get_library_root_logger()
    example_logger = pfnopt.logging.get_logger('pfnopt.bar')

    # Default handler enabled
    pfnopt.logging.enable_default_handler()
    assert library_root_logger.handlers
    example_logger.warning('hey')
    _, err = capsys.readouterr()
    assert 'hey' in err

    # Default handler disabled
    pfnopt.logging.disable_default_handler()
    assert not library_root_logger.handlers
    example_logger.warning('yoyo')
    _, err = capsys.readouterr()
    assert 'yoyo' not in err


def test_verbosity(capsys):
    # type: (_pytest.capture.CaptureFixture) -> None

    # We need to reconstruct our default handler to properly capture stderr.
    pfnopt.logging._reset_library_root_logger()
    library_root_logger = pfnopt.logging._get_library_root_logger()
    example_logger = pfnopt.logging.get_logger('pfnopt.hoge')
    pfnopt.logging.enable_default_handler()

    # level INFO
    pfnopt.logging.set_verbosity(pfnopt.logging.INFO)
    assert library_root_logger.getEffectiveLevel() == logging.INFO
    example_logger.warning('hello-warning')
    example_logger.info('hello-info')
    example_logger.debug('hello-debug')
    _, err = capsys.readouterr()
    assert 'hello-warning' in err
    assert 'hello-info' in err
    assert 'hello-debug' not in err

    # level WARNING
    pfnopt.logging.set_verbosity(pfnopt.logging.WARNING)
    assert library_root_logger.getEffectiveLevel() == logging.WARNING
    example_logger.warning('bye-warning')
    example_logger.info('bye-info')
    example_logger.debug('bye-debug')
    _, err = capsys.readouterr()
    assert 'bye-warning' in err
    assert 'bye-info' not in err
    assert 'bye-debug' not in err
