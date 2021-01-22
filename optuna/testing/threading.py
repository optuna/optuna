import threading
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple


class _TestableThread(threading.Thread):
    def __init__(self, target: Callable[..., Any], args: Tuple):
        threading.Thread.__init__(self, target=target, args=args)
        self.exc: Optional[BaseException] = None

    def run(self) -> None:
        try:
            threading.Thread.run(self)
        except BaseException as e:
            self.exc = e

    def join(self, timeout: Optional[float] = None) -> None:
        super(_TestableThread, self).join(timeout)
        if self.exc:
            raise self.exc
