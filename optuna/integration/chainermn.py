from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
import warnings

from optuna import TrialPruned
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.study import Study
from optuna.trial import BaseTrial
from optuna.trial import Trial
from optuna.integration.mpi import MPIStudy
from optuna.integration.mpi import MPITrial
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna import TrialPruned


with try_import() as _imports:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA


class _ChainerMNObjectiveFunc(object):
    """A wrapper of an objective function to incorporate Optuna with ChainerMN.

    Note that this class is not supposed to be used by library users.

    Args:
        func:
            A callable that implements objective function.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(
        self,
        func: Callable[["ChainerMNTrial", "CommunicatorBase"], float],
        comm: "CommunicatorBase",
    ) -> None:

        self.comm = comm
        self.objective = func

    def __call__(self, trial: Trial) -> float:

        self.comm.mpi_comm.bcast(True)
        return self.objective(ChainerMNTrial(trial, self.comm), self.comm)


class ChainerMNStudy(MPIStudy):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNStudy` provides the same interface as
        :class:`~optuna.study.Study`. Please refer to :class:`optuna.study.Study` for further
        details.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/chainermn_integration.py>`__
    if you want to optimize an objective function that trains neural network
    written with ChainerMN.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(self, study: Study, comm: "CommunicatorBase") -> None:

        _imports.check()

        if isinstance(study._storage, InMemoryStorage):
            raise ValueError("ChainerMN integration is not available with InMemoryStorage.")

        if isinstance(study._storage, RDBStorage):
            if study._storage.engine.dialect.name == "sqlite":
                warnings.warn(
                    "SQLite may cause synchronization problems when used with "
                    "ChainerMN integration. Please use other DBs like PostgreSQL."
                )

        study_names = comm.mpi_comm.allgather(study.study_name)
        if len(set(study_names)) != 1:
            raise ValueError("Please make sure an identical study name is shared among workers.")

        super(MPIStudy, self).__setattr__("delegate", study)
        super(MPIStudy, self).__setattr__("comm", comm)

    def optimize(
        self,
        func: Callable[["ChainerMNTrial", "CommunicatorBase"], float],
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        catch: Tuple[Type[Exception], ...] = (),
    ) -> None:
        """Optimize an objective function.

        This method provides the same interface as :func:`optuna.study.Study.optimize` except
        the absence of ``n_jobs`` argument.
        """

        if self.comm.mpi_comm.rank == 0:
            func_mn = _ChainerMNObjectiveFunc(func, self.comm)
            try:
                self.delegate.optimize(func_mn, n_trials=n_trials, timeout=timeout, catch=catch)
            finally:
                self.comm.mpi_comm.bcast(False)
        else:
            has_next_trial = self.comm.mpi_comm.bcast(None)
            while True:
                if not has_next_trial:
                    break
                try:
                    func(ChainerMNTrial(None, self.comm), self.comm)

                    # We assume that if a node raises an exception,
                    # all other nodes will do the same.
                    #
                    # The responsibility to handle acceptable exceptions (i.e., `TrialPruned` and
                    # `catch`) is in the rank-0 node, so other nodes simply ignore them.
                except TrialPruned:
                    pass
                except catch:
                    pass
                finally:
                    has_next_trial = self.comm.mpi_comm.bcast(None)


class ChainerMNTrial(MPITrial):
    """A wrapper of :class:`~optuna.trial.Trial` to incorporate Optuna with ChainerMN.

    .. seealso::
        :class:`~optuna.integration.chainermn.ChainerMNTrial` provides the same interface as
        :class:`~optuna.trial.Trial`. Please refer to :class:`optuna.trial.Trial` for further
        details.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object if the caller is rank0 worker,
            :obj:`None` otherwise.
        comm:
            A `ChainerMN communicator <https://docs.chainer.org/en/stable/chainermn/reference/
            index.html#communicators>`_.
    """

    def __init__(self, trial: Optional[Trial], comm: "CommunicatorBase") -> None:

        self.delegate = trial
        self.comm = comm.mpi_comm
