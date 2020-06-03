from types import TracebackType
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union


class _DeferredExceptionContextManager(object):
    """Context manager to defer exceptions.

    Args:
        catch:
            Exception types to defer.
        message:
            Message to include in deferred exception.

    """

    def __init__(self, catch: Union[Tuple[()], Tuple[Type[Exception], ...]], message: str) -> None:
        self._catch = catch
        self._message = message
        self._caught_exception = None  # type: Optional[Tuple[Type[Exception], Exception]]

    def __enter__(self) -> "_DeferredExceptionContextManager":
        """Enter the context manager.

        Returns:
            Itself.

        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context manager.

        Args:
            exc_type:
                Raised exception type. :obj:`None` if nothing is raised.
            exc_value:
                Raised exception object. :obj:`None` if nothing is raised.
            traceback:
                Associated traceback. :obj:`None` if nothing is raised.

        Returns:
            :obj:`None` if nothing was caught, otherwise :obj:`True`.
            :obj:`True` will suppress any exceptions avoiding them from propagating.

        """
        if isinstance(exc_value, self._catch):
            assert exc_type is not None
            assert exc_value is not None
            self._caught_exception = (exc_type, exc_value)
            return True
        return None

    def is_successful(self) -> bool:
        """Return whether the context manager has caught any exceptions.

        Returns:
            :obj:`True` if no exceptions are caught, :obj:`False` otherwise.

        """
        return self._caught_exception is None

    def check(self) -> None:
        """Check whether the context manger has caught any exceptions.

        Raises:
            :exc:`Exception`:
                If any exception was caught, raises the caught exception with a modified message.

        """
        if self._caught_exception is not None:
            exc_type, exc_value = self._caught_exception
            raise exc_type(self._message) from exc_value


def try_import(
    catch: Union[Tuple[()], Tuple[Type[Exception], ...]] = (ImportError,),
) -> _DeferredExceptionContextManager:
    """Create a context manager that can wrap imports of optional packages to defer exceptions.

    Args:
        catch:
            Exception types to defer.

    """
    return _DeferredExceptionContextManager(
        catch, message="Failed to import an optional package. See causing exceptions for details."
    )
