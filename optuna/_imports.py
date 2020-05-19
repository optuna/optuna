from types import TracebackType
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union


class _DeferredExceptionContextManager(object):
    def __init__(
        self, catch: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]], message: str
    ) -> None:
        # TODO(hvy): Consider making `message` an optional argument.
        self._catch = catch
        self._message = message
        self._caught_exception = None  # type: Optional[Tuple[type, Exception]]

    def __enter__(self) -> "_DeferredExceptionContextManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        if isinstance(exc_value, self._catch):
            assert exc_type is not None
            assert exc_value is not None
            self._caught_exception = (exc_type, exc_value)
            return True  # Suppress and defer the caught error.
        return None

    def is_successful(self) -> bool:
        return self._caught_exception is None

    def check(self) -> None:
        if self._caught_exception is not None:
            exc_type, exc_value = self._caught_exception
            raise exc_type(
                "{message}{exc_value}".format(message=self._message, exc_value=exc_value,)
            )


def try_import(
    message: Optional[str] = None,
    catch: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]] = (ImportError,),
) -> _DeferredExceptionContextManager:
    return _DeferredExceptionContextManager(catch, "Failed to import an optional package. ",)
