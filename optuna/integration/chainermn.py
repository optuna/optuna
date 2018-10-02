from __future__ import absolute_import

from typing import Callable  # NOQA
from typing import Optional  # NOQA

from optuna.pruners import BasePruner  # NOQA
from optuna.samplers import BaseSampler  # NOQA
from optuna.storages import InMemoryStorage
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA

try:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class ObjectiveFuncChainerMN(object):
    def __init__(self, func, comm):
        # type: (Callable[[Trial, CommunicatorBase], float], CommunicatorBase) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.mpi_comm.bcast((True, trial.trial_id))
        return self.objective(trial, self.comm)


def minimize_chainermn(
        func,  # type: Callable[[Trial, CommunicatorBase], float]
        study,  # type: Study
        comm,  # type: CommunicatorBase
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        sampler=None,  # type: BaseSampler
        pruner=None,  # type: BasePruner
):
    # type: (...) -> Study

    _check_chainermn_availability()

    if sampler is not None:
        study.sampler = sampler
    if pruner is not None:
        study.pruner = pruner

    if isinstance(study.storage, InMemoryStorage):
        raise ValueError('ChainerMN integration is not available with InMemoryStorage.')

    study_names = comm.mpi_comm.allgather(study.study_name)
    if len(set(study_names)) != 1:
        raise ValueError('Please make sure an identical study name is shared among workers.')

    if comm.rank == 0:
        study.run(
            ObjectiveFuncChainerMN(func, comm),
            n_trials=n_trials, timeout_seconds=timeout, n_jobs=1)
        comm.mpi_comm.bcast((False, None))
    else:
        while True:
            has_next_trial, trial_id = comm.mpi_comm.bcast(None)
            if not has_next_trial:
                break
            trial = Trial(study, trial_id)
            func(trial, comm)

    return study


def _check_chainermn_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'ChainerMN is not available. Please install ChainerMN to use this feature. '
            'ChainerMN can be installed by executing `$ pip install chainermn`. '
            'For further information, please refer to the installation guide of ChainerMN. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
