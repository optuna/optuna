try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False
