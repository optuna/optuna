import threading
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple


class _TestableThread(threading.Thread):
    """Wrapper around `threading.Thread` that propagates exceptions."""

    def __init__(self, target: Callable[..., Any], args: Tuple):
        threading.Thread.__init__(self, target=target, args=args)
        self.exc: Optional[BaseException] = None

    def run(self) -> None:
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.
        """
        try:
            threading.Thread.run(self)
        except BaseException as e:
            self.exc = e

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait until the thread terminates.

        This blocks the calling thread until the thread whose join() method is
        called terminates -- either normally or through an unhandled exception
        or until the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof). As join() always returns None, you must call
        is_alive() after join() to decide whether a timeout happened -- if the
        thread is still alive, the join() call timed out.

        When the timeout argument is not present or None, the operation will
        block until the thread terminates.

        A thread can be join()ed many times.

        join() raises a RuntimeError if an attempt is made to join the current
        thread as that would cause a deadlock. It is also an error to join() a
        thread before it has been started and attempts to do so raises the same
        exception.
        """
        super(_TestableThread, self).join(timeout)
        if self.exc:
            raise self.exc
