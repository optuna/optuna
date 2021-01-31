from optuna.exceptions import TrialPruned
from datetime import datetime
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
import warnings

import torch
import torch.distributed as dist

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType


class DDPTrial(optuna.trial.BaseTrial):
    def __init__(self, trial: Optional[optuna.trial.Trial]) -> None:
        self.delegate = trial
        pass

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_float(name, low, high, step=step, log=log)

        return self._call_and_communicate(func, torch.float)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_uniform(name, low, high)

        return self._call_and_communicate(func, torch.float)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_loguniform(name, low, high)

        return self._call_and_communicate(func, torch.float)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_discrete_uniform(name, low, high, q=q)

        return self._call_and_communicate(func, torch.float)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_int(name, low, high, step=step, log=log)

        return self._call_and_communicate(func, torch.int)

    def suggest_categorical(self, name: str, choices: Sequence["CategoricalChoiceType"]) -> Any:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.suggest_categorical(name, choices)

        return self._call_and_communicate_obj(func)

    def report(self, value: float, step: int) -> None:

        if dist.get_rank() == 0:
            assert self.delegate is not None
            self.delegate.report(value, step)
        dist.barrier()

    def should_prune(self) -> bool:
        def func() -> bool:

            assert self.delegate is not None
            return self.delegate.should_prune()

        return self._call_and_communicate(func, torch.bool)

    def set_user_attr(self, key: str, value: Any) -> None:

        if dist.get_rank() == 0:
            assert self.delegate is not None
            self.delegate.set_user_attr(key, value)
        dist.barrier()

    def set_system_attr(self, key: str, value: Any) -> None:

        if dist.get_rank() == 0:
            assert self.delegate is not None
            self.delegate.set_system_attr(key, value)
        dist.barrier()

    @property
    def number(self) -> int:
        def func() -> int:

            assert self.delegate is not None
            return self.delegate.number

        return self._call_and_communicate(func, torch.int)

    @property
    def trial_id(self) -> int:

        warnings.warn(
            "The use of `MPITrial.trial_id` is deprecated. "
            "Please use `MPITrial.number` instead.",
            DeprecationWarning,
        )
        return self._trial_id

    @property
    def _trial_id(self) -> int:
        def func() -> int:

            assert self.delegate is not None
            return self.delegate._trial_id

        return self._call_and_communicate(func, torch.int)

    @property
    def params(self) -> Dict[str, Any]:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.params

        return self._call_and_communicate_obj(func)

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.distributions

        return self._call_and_communicate_obj(func)

    @property
    def user_attrs(self) -> Dict[str, Any]:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.user_attrs

        return self._call_and_communicate_obj(func)

    @property
    def system_attrs(self) -> Dict[str, Any]:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.system_attrs

        return self._call_and_communicate_obj(func)

    @property
    def datetime_start(self) -> Optional[datetime]:
        def func() -> float:

            assert self.delegate is not None
            return self.delegate.datetime_start

        return self._call_and_communicate_obj(func)

    def _call_and_communicate(self, func: Callable, dtype) -> Any:
        buffer = torch.empty(1, dtype=dtype)
        if dist.get_rank() == 0:
            result = func()
            buffer[0] = result
        if dist.get_backend() == "nccl":
            buffer = buffer.cuda()
        dist.broadcast(buffer, src=0)
        return buffer.cpu().numpy().tolist()[0]

    def _call_and_communicate_obj(self, func: Callable) -> Any:
        buffer = None
        size_buffer = torch.empty(1, dtype=torch.int)
        if dist.get_rank() == 0:
            result = func()
            buffer = to_tensor(result)
            size_buffer[0] = buffer.shape[0]
        if dist.get_backend() == "nccl":
            size_buffer = size_buffer.cuda()
        dist.broadcast(size_buffer, src=0)
        buffer_size = size_buffer.cpu().numpy().tolist()[0]
        if dist.get_rank() != 0:
            buffer = torch.empty(buffer_size, dtype=torch.uint8)
        assert buffer is not None
        if dist.get_backend() == "nccl":
            buffer = buffer.cuda()
        dist.broadcast(buffer, src=0)
        return from_tensor(buffer)


def to_tensor(obj: Any) -> torch.Tensor:
    b = bytearray(pickle.dumps(obj))
    return torch.tensor(b, dtype=torch.uint8)


def from_tensor(tensor: torch.Tensor) -> Any:
    b = bytearray(tensor.to("cpu").numpy().tolist())
    return pickle.loads(b)
