from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union


import optuna
from optuna._experimental import experimental
from optuna import mo
from optuna import logging
from optuna.storages import BaseStorage
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.study import Study
from optuna.trial import Trial


ObjectiveFuncType = Callable[["mo.trial.MoTrial"], List[float]]
CallbackFuncType = Callable[["mo.study.MoStudy", "mo.trial.FrozenMoTrial"], None]

_logger = logging.get_logger(__name__)


@experimental("1.14.0")
def create_mo_study(
    n_objectives: int,
    study_name: Optional[str] = None,
    storage: Union[None, str, BaseStorage] = None,
    sampler: Optional["mo.samplers.BaseMoSampler"] = None,
    directions: Optional[List[str]] = None,
    load_if_exists: bool = False,
):
    # TODO(ohta): Support pruner.
    mo_sampler = sampler or mo.samplers.RandomMoSampler()
    sampler = mo.samplers._MoSamplerAdapter(mo_sampler)

    if directions is None:
        directions = ["minimize" for _ in range(n_objectives)]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=load_if_exists,
    )

    study.set_system_attr("mo.study.n_objectives", n_objectives)
    study.set_system_attr("mo.study.directions", directions)

    return MoStudy(study)


@experimental("1.14.0")
def load_mo_study(
    study_name: str,
    storage: Union[str, BaseStorage],
    sampler: Optional["mo.samplers.BaseMoSampler"] = None,
):
    mo_sampler = sampler or mo.samplers.RandomSampler()
    sampler = mo.samplers._MoSamplerAdapter(mo_sampler)

    study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler)

    return MoStudy(study)


@experimental("1.14.0")
class MoStudy(object):
    def __init__(self, study: Study):
        self._study = study
        self._n_objectives = study.system_attrs["mo.study.n_objectives"]

        self._directions = []
        for d in study.system_attrs["mo.study.directions"]:
            if d == "minimize":
                self._directions.append(StudyDirection.MINIMIZE)
            elif d == "maximize":
                self._directions.append(StudyDirection.MAXIMIZE)
            else:
                raise ValueError("Unknown direction ({}) is specified.".format(d))

        if self._n_objectives < 1:
            raise ValueError("The number of objectives must be greater than 0.")

        if self._n_objectives != len(self._directions):
            raise ValueError("Objective and direction numbers don't match.")

        self._study._log_completed_trial = _log_completed_trial

    @property
    def n_objectives(self) -> int:
        return self._n_objectives

    @property
    def directions(self) -> List[StudyDirection]:
        return self._directions

    def optimize(
        self,
        objective: ObjectiveFuncType,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        n_jobs: int = 1,
        catch: Union[Tuple[()], Tuple[Type[Exception]]] = (),
        callbacks: Optional[List[CallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        def mo_objective(trial: Trial) -> float:
            mo_trial = mo.trial.MoTrial(trial)
            values = objective(mo_trial)
            mo_trial._report_complete_values(values)
            return 0.0  # Dummy value.

        self._study.optimize(
            mo_objective,
            timeout=timeout,
            n_trials=n_trials,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._study.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._study.system_attrs

    def set_user_attr(self, key: str, value: Any) -> None:
        self._study.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any):
        self._study.set_system_attr(key, value)

    def enqueue_trial(self, params: Dict[str, Any]) -> None:
        self._study.enqueue_trial(params)

    @property
    def trials(self) -> List["mo.trial.FrozenMoTrial"]:
        return self.get_trials()

    def get_trials(self, deepcopy: bool = True) -> List["mo.trial.FrozenMoTrial"]:
        return [
            mo.trial.FrozenMoTrial(self.n_objectives, t)
            for t in self._study.get_trials(deepcopy=deepcopy)
        ]

    @property
    def pareto_front_trials(self) -> List["mo.trial.FrozenMoTrial"]:
        pareto_front = []
        trials = [t for t in self.trials if t.state == TrialState.COMPLETE]

        # TODO(ohta): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
        for trial in trials:
            dominated = False
            for other in trials:
                if other._dominates(trial):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(trial)

        return pareto_front


def _log_completed_trial(trial: Trial, result: float) -> None:
    values = mo.trial.MoTrial(trial)._values
    _logger.info(
        "Finished trial#{} with values: {} with parameters: {}.".format(
            trial.number, values, trial.params,
        )
    )
