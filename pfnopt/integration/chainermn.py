from __future__ import absolute_import

from typing import Optional  # NOQA
from typing import Union  # NOQA

from pfnopt.pruners import BasePruner  # NOQA
from pfnopt.samplers import BaseSampler  # NOQA
from pfnopt.storages import BaseStorage  # NOQA
from pfnopt.storages import InMemoryStorage
from pfnopt.study import get_study
from pfnopt.study import minimize
from pfnopt.study import ObjectiveFuncType  # NOQA
from pfnopt.study import Study  # NOQA
from pfnopt.trial import Trial  # NOQA

try:
    from chainermn.communicators.communicator_base import CommunicatorBase  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class ObjectiveFuncChainerMN(object):
    def __init__(self, func, comm):
        # type: (ObjectiveFuncType, CommunicatorBase) -> None

        self.comm = comm
        self.objective = func

    def __call__(self, trial):
        # type: (Trial) -> float

        self.comm.mpi_comm.bcast((True, trial.trial_id))
        return self.objective(trial)


def minimize_chainermn(
        func,  # type: ObjectiveFuncType
        study,  # type: Union[str, Study]
        comm,  # type: CommunicatorBase
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        storage=None,  # type: Union[None, str, BaseStorage]
        sampler=None,  # type: BaseSampler
        pruner=None,  # type: BasePruner
):
    # type: (...) -> Study

    _check_chainermn_availability()

    study = get_study(study=study, storage=storage, sampler=sampler, pruner=pruner)

    if isinstance(study.storage, InMemoryStorage):
        raise ValueError('ChainerMN integration is not available with InMemoryStorage.')

    study_uuids = comm.mpi_comm.allgather(study.study_uuid)
    if len(set(study_uuids)) != 1:
        raise ValueError('Please make sure an identical study UUID is shared among workers.')

    if comm.rank == 0:
        minimize(
            ObjectiveFuncChainerMN(func, comm),
            n_trials=n_trials, timeout=timeout, n_jobs=1, study=study)
        comm.mpi_comm.bcast((False, None))
    else:
        while True:
            has_next_trial, trial_id = comm.mpi_comm.bcast(None)
            if not has_next_trial:
                break
            trial = Trial(study, trial_id)
            func(trial)

    return study


def _check_chainermn_availability():
    if not _available:
        raise ImportError(
            'ChainerMN is not available. Please install ChainerMN to use this feature. '
            'ChainerMN can be installed by executing `$ pip install chainermn`. '
            'For further information, please refer to the installation guide of ChainerMN. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
