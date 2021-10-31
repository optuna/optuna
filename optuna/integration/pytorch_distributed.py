from datetime import datetime
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import optuna
from optuna._deprecated import deprecated
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType


with try_import() as _imports:
    import torch
    import torch.distributed as dist


_suggest_deprecated_msg = (
    "Use :func:`~optuna.integration.TorchDistributedTrial.suggest_float` instead."
)


@experimental("2.6.0")
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

        if dist.get_rank() == 0:
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

    @deprecated("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high)

    @deprecated("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high, log=True)

    @deprecated("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        return self.suggest_float(name, low, high, step=q)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        def func() -> float:

            assert self._delegate is not None
            return self._delegate.suggest_int(name, low, high, step=step, log=log)

        return self._call_and_communicate(func, torch.int)

    def suggest_categorical(self, name: str, choices: Sequence["CategoricalChoiceType"]) -> Any:
        def func() -> CategoricalChoiceType:

            assert self._delegate is not None
            return self._delegate.suggest_categorical(name, choices)

        return self._call_and_communicate_obj(func)

    def report(self, value: float, step: int) -> None:
        err = None
        if dist.get_rank() == 0:
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

    def should_prune(self) -> bool:
        def func() -> bool:

            assert self._delegate is not None
            # Some pruners return numpy.bool_, which is incompatible with bool.
            return bool(self._delegate.should_prune())

        # torch.bool seems to be the correct type, but the communication fails
        # due to the RuntimeError.
        return self._call_and_communicate(func, torch.uint8)

    def set_user_attr(self, key: str, value: Any) -> None:
        err = None
        if dist.get_rank() == 0:
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

    def set_system_attr(self, key: str, value: Any) -> None:
        err = None

        if dist.get_rank() == 0:
            try:
                assert self._delegate is not None
                self._delegate.set_system_attr(key, value)
            except Exception as e:
                err = e
            err = self._broadcast(err)
        else:
            err = self._broadcast(err)

        if err is not None:
            raise err

    @property
    def number(self) -> int:
        def func() -> int:

            assert self._delegate is not None
            return self._delegate.number

        return self._call_and_communicate(func, torch.int)

    @property
    def params(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self._delegate is not None
            return self._delegate.params

        return self._call_and_communicate_obj(func)

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        def func() -> Dict[str, BaseDistribution]:

            assert self._delegate is not None
            return self._delegate.distributions

        return self._call_and_communicate_obj(func)

    @property
    def user_attrs(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self._delegate is not None
            return self._delegate.user_attrs

        return self._call_and_communicate_obj(func)

    @property
    def system_attrs(self) -> Dict[str, Any]:
        def func() -> Dict[str, Any]:

            assert self._delegate is not None
            return self._delegate.system_attrs

        return self._call_and_communicate_obj(func)

    @property
    def datetime_start(self) -> Optional[datetime]:
        def func() -> Optional[datetime]:

            assert self._delegate is not None
            return self._delegate.datetime_start

        return self._call_and_communicate_obj(func)

    def _call_and_communicate(self, func: Callable, dtype: "torch.dtype") -> Any:
        buffer = torch.empty(1, dtype=dtype)
        rank = dist.get_rank()
        if rank == 0:
            result = func()
            buffer[0] = result
        if self._device is not None:
            buffer = buffer.to(self._device)
        dist.broadcast(buffer, src=0)
        return buffer.item()

    def _call_and_communicate_obj(self, func: Callable) -> Any:
        rank = dist.get_rank()
        result = func() if rank == 0 else None
        return self._broadcast(result)

    def _broadcast(self, value: Optional[Any]) -> Any:
        buffer = None
        size_buffer = torch.empty(1, dtype=torch.int)
        rank = dist.get_rank()
        if rank == 0:
            buffer = _to_tensor(value)
            size_buffer[0] = buffer.shape[0]
        if self._device is not None:
            size_buffer = size_buffer.to(self._device)
        dist.broadcast(size_buffer, src=0)
        buffer_size = int(size_buffer.item())
        if rank != 0:
            buffer = torch.empty(buffer_size, dtype=torch.uint8)
        assert buffer is not None
        if self._device is not None:
            buffer = buffer.to(self._device)
        dist.broadcast(buffer, src=0)
        return _from_tensor(buffer)


def _to_tensor(obj: Any) -> "torch.Tensor":
    b = bytearray(pickle.dumps(obj))
    return torch.tensor(b, dtype=torch.uint8)


def _from_tensor(tensor: "torch.Tensor") -> Any:
    b = bytearray(tensor.tolist())
    return pickle.loads(b)
