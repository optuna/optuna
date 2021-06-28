import threading
from typing import Any
from typing import Optional


class _TestableThread(threading.Thread):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.exc: Optional[BaseException] = None

    def run(self) -> None:
        try:
            super().run()
        except BaseException as e:
            self.exc = e

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self.exc:
            raise self.exc
