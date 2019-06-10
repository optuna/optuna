import tempfile

import optuna
from optuna import types

if types.TYPE_CHECKING:
    from types import TracebackType  # NOQA
    from typing import Any  # NOQA
    from typing import IO  # NOQA
    from typing import Optional  # NOQA
    from typing import Type  # NOQA

SQLITE3_TIMEOUT = 300


class StorageSupplier(object):

    _common_tempfile = None  # type: Optional[IO[Any]]

    def __init__(self, storage_specifier, enable_cache=True):
        # type: (str, bool) -> None

        self.storage_specifier = storage_specifier
        self.tempfile = None  # type: Optional[IO[Any]]
        self.enable_cache = enable_cache

    def __enter__(self):
        # type: () -> Optional[optuna.storages.BaseStorage]

        if self.storage_specifier == 'none':
            return None
        elif self.storage_specifier == 'new':
            self.tempfile = tempfile.NamedTemporaryFile()
            url = 'sqlite:///{}'.format(self.tempfile.name)
            return optuna.storages.RDBStorage(
                url,
                engine_kwargs={'connect_args': {'timeout': SQLITE3_TIMEOUT}},
                enable_cache=self.enable_cache,
            )
        elif self.storage_specifier == 'common':
            assert self._common_tempfile is not None
            url = 'sqlite:///{}'.format(self._common_tempfile.name)
            return optuna.storages.RDBStorage(
                url,
                engine_kwargs={'connect_args': {'timeout': SQLITE3_TIMEOUT}},
                enable_cache=self.enable_cache,
            )
        else:
            assert False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        if self.tempfile:
            self.tempfile.close()

    @classmethod
    def setup_common_tempfile(cls):
        # type: () -> None

        cls._common_tempfile = tempfile.NamedTemporaryFile()

    @classmethod
    def teardown_common_tempfile(cls):
        # type: () -> None

        assert cls._common_tempfile is not None
        cls._common_tempfile.close()
