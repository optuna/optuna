from datetime import datetime
import functools
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar

import optuna
from optuna._deprecated import deprecated_func
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType


with try_import() as _imports:
    import torch
    import torch.distributed as dist


if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _T = TypeVar("_T")
    _P = ParamSpec("_P")


_suggest_deprecated_msg = (
    "Use :func:`~optuna.integration.TorchDistributedTrial.suggest_float` instead."
)


def broadcast_properties(f: "Callable[_P, _T]") -> "Callable[_P, _T]":
    """Method decorator to fetch updated trial properties from rank 0 after ``f`` is run.

    This decorator ensures trial properties (params, distributions, etc.) on all distributed
    processes are up-to-date with the wrapped trial stored on rank 0.
    It should be applied to all :class:`~optuna.integration.TorchDistributedTrial`
    methods that update property values.
    """

    @functools.wraps(f)
    def wrapped(*args: "_P.args", **kwargs: "_P.kwargs") -> "_T":
        # TODO(nlgranger): Remove type ignore after mypy includes
        # https://github.com/python/mypy/pull/12668
        self: TorchDistributedTrial = args[0]  # type: ignore

        def fetch_properties() -> Sequence:
            assert self._delegate is not None
            return (
                self._delegate.number,
                self._delegate.params,
                self._delegate.distributions,
                self._delegate.user_attrs,
                self._delegate.system_attrs,
                self._delegate.datetime_start,
            )

        try:
            return f(*args, **kwargs)
        finally:
            (
                self._number,
                self._params,
                self._distributions,
                self._user_attrs,
                self._system_attrs,
                self._datetime_start,
            ) = self._call_and_communicate_obj(fetch_properties)

    return wrapped


@experimental_class("2.6.0")
class TorchDistributedTrial(optuna.trial.BaseTrial):
    """A wrapper of :class:`~optuna.trial.Trial` to incorporate Optuna with PyTorch distributed.

    .. seealso::
        :class:`~optuna.integration.TorchDistributedTrial` provides the same interface as
        :class:`~optuna.trial.Trial`. Please refer to :class:`optuna.trial.Trial` for further
        details.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    pytorch/pytorch_distributed_simple.py>`__
    if you want to optimize an objective function that trains neural network
    written with PyTorch distributed data parallel.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object or :obj:`None`. Please set trial object in
            rank-0 node and set :obj:`None` in the other rank node.
        device:
            A `torch.device` to communicate with the other nodes. Please set a CUDA device
            assigned to the current node if you use "nccl" as `torch.distributed` backend.

    .. note::
        The methods of :class:`~optuna.integration.TorchDistributedTrial` are expected to be
        called by all workers at once. They invoke synchronous data transmission to share
        processing results and synchronize timing.

    """

    def __init__(
        self, trial: Optional[optuna.trial.Trial], device: Optional["torch.device"] = None
    ) -> None:

        _imports.check()

        if dist.get_rank() == 0:  # type: ignore
            if not isinstance(trial, optuna.trial.Trial):
                raise ValueError(
                    "Rank 0 node expects an optuna.trial.Trial instance as the trial argument."
                )
        else:
            if trial is not None:
                raise ValueError(
                    "Non-rank 0 node is supposed to recieve None as the trial argument."
                )

            assert trial is None, "error message"
        self._delegate = trial
        self._device = device

        self._number = self._broadcast(getattr(self._delegate, "number", None))
        self._params = self._broadcast(getattr(self._delegate, "params", None))
        self._distributions = self._broadcast(getattr(self._delegate, "distributions", None))
        self._user_attrs = self._broadcast(getattr(self._delegate, "user_attrs", None))
        self._system_attrs = self._broadcast(getattr(self._delegate, "system_attrs", None))
        self._datetime_start = self._broadcast(getattr(self._delegate, "datetime_start", None))

    @broadcast_properties
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        def func() -> float:

            assert self._delegate is not None
            return self._delegate.suggest_float(name, low, high, step=step, log=log)

        return self._call_and_communicate(func, torch.float)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high, log=True)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        return self.suggest_float(name, low, high, step=q)

    @broadcast_properties
    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        def func() -> float:

            assert self._delegate is not None
            return self._delegate.suggest_int(name, low, high, step=step, log=log)

        return self._call_and_communicate(func, torch.int)

    @broadcast_properties
    def suggest_categorical(self, name: str, choices: Sequence["CategoricalChoiceType"]) -> Any:
        def func() -> CategoricalChoiceType:

            assert self._delegate is not None
            return self._delegate.suggest_categorical(name, choices)

        return self._call_and_communicate_obj(func)

    @broadcast_properties
    def report(self, value: float, step: int) -> None:
        err = None
        if dist.get_rank() == 0:  # type: ignore
            try:
                assert self._delegate is not None
                self._delegate.report(value, step)
            except Exception as e:
                err = e
            err = self._broadcast(err)
        else:
            err = self._broadcast(err)

        if err is not None:
            raise err

    @broadcast_properties
    def should_prune(self) -> bool:
        def func() -> bool:

            assert self._delegate is not None
            # Some pruners return numpy.bool_, which is incompatible with bool.
            return bool(self._delegate.should_prune())

        # torch.bool seems to be the correct type, but the communication fails
        # due to the RuntimeError.
        return self._call_and_communicate(func, torch.uint8)

    @broadcast_properties
    def set_user_attr(self, key: str, value: Any) -> None:
        err = None
        if dist.get_rank() == 0:  # type: ignore
            try:
                assert self._delegate is not None
                self._delegate.set_user_attr(key, value)
            except Exception as e:
                err = e
            err = self._broadcast(err)
        else:
            err = self._broadcast(err)

        if err is not None:
            raise err

    @broadcast_properties
    @deprecated_func("3.1.0", "6.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:
        err = None

        if dist.get_rank() == 0:  # type: ignore
            try:
                assert self._delegate is not None
                self._delegate.storage.set_trial_system_attr(self._delegate._trial_id, key, value)
            except Exception as e:
                err = e
            err = self._broadcast(err)
        else:
            err = self._broadcast(err)

        if err is not None:
            raise err

    @property
    def number(self) -> int:
        return self._number

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        return self._distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._system_attrs

    @property
    def datetime_start(self) -> Optional[datetime]:
        return self._datetime_start

    def _call_and_communicate(self, func: Callable, dtype: "torch.dtype") -> Any:
        buffer = torch.empty(1, dtype=dtype)
        rank = dist.get_rank()  # type: ignore
        if rank == 0:
            result = func()
            buffer[0] = result
        if self._device is not None:
            buffer = buffer.to(self._device)
        dist.broadcast(buffer, src=0)  # type: ignore
        return buffer.item()

    def _call_and_communicate_obj(self, func: Callable) -> Any:
        rank = dist.get_rank()  # type: ignore
        result = func() if rank == 0 else None
        return self._broadcast(result)

    def _broadcast(self, value: Optional[Any]) -> Any:
        buffer = None
        size_buffer = torch.empty(1, dtype=torch.int)
        rank = dist.get_rank()  # type: ignore
        if rank == 0:
            buffer = _to_tensor(value)
            size_buffer[0] = buffer.shape[0]
        if self._device is not None:
            size_buffer = size_buffer.to(self._device)
        dist.broadcast(size_buffer, src=0)  # type: ignore
        buffer_size = int(size_buffer.item())
        if rank != 0:
            buffer = torch.empty(buffer_size, dtype=torch.uint8)
        assert buffer is not None
        if self._device is not None:
            buffer = buffer.to(self._device)
        dist.broadcast(buffer, src=0)  # type: ignore
        return _from_tensor(buffer)


def _to_tensor(obj: Any) -> "torch.Tensor":
    b = bytearray(pickle.dumps(obj))
    return torch.tensor(b, dtype=torch.uint8)


def _from_tensor(tensor: "torch.Tensor") -> Any:
    b = bytearray(tensor.tolist())
    return pickle.loads(b)
