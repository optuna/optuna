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
from optuna import logging
from optuna import multi_objective
from optuna.storages import BaseStorage
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.study import Study
from optuna.trial import Trial


ObjectiveFuncType = Callable[["multi_objective.trial.MultiObjectiveTrial"], List[float]]
CallbackFuncType = Callable[
    [
        "multi_objective.study.MultiObjectiveStudy",
        "multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]

_logger = logging.get_logger(__name__)


@experimental("1.4.0")
def create_study(
    directions: List[str],
    study_name: Optional[str] = None,
    storage: Union[None, str, BaseStorage] = None,
    sampler: Optional["multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    load_if_exists: bool = False,
):
    # TODO(ohta): Support pruner.
    mo_sampler = sampler or multi_objective.samplers.RandomMultiObjectiveSampler()
    sampler = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)

    if not isinstance(directions, list):
        raise ValueError("`directions` must be a list.")

    if not all(d in ["minimize", "maximize"] for d in directions):
        raise ValueError("`directions` includes unknown direction names.")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=load_if_exists,
    )

    study.set_system_attr("multi_objective.study.directions", directions)

    return MultiObjectiveStudy(study)


@experimental("1.4.0")
def load_study(
    study_name: str,
    storage: Union[str, BaseStorage],
    sampler: Optional["multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
):
    mo_sampler = sampler or multi_objective.samplers.RandomSampler()
    sampler = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)

    study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)

    return MultiObjectiveStudy(study)


@experimental("1.4.0")
class MultiObjectiveStudy(object):
    def __init__(self, study: Study):
        self._study = study

        self._directions = []
        for d in study.system_attrs["multi_objective.study.directions"]:
            if d == "minimize":
                self._directions.append(StudyDirection.MINIMIZE)
            elif d == "maximize":
                self._directions.append(StudyDirection.MAXIMIZE)
            else:
                raise ValueError("Unknown direction ({}) is specified.".format(d))
        self._n_objectives = len(self._directions)

        if self._n_objectives < 1:
            raise ValueError("The number of objectives must be greater than 0.")

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
            mo_trial = multi_objective.trial.MultiObjectiveTrial(trial)
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
    def trials(self) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        return self.get_trials()

    def get_trials(
        self, deepcopy: bool = True
    ) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        return [
            multi_objective.trial.FrozenMultiObjectiveTrial(self.n_objectives, t)
            for t in self._study.get_trials(deepcopy=deepcopy)
        ]

    @property
    def pareto_front_trials(self) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        pareto_front = []
        trials = [t for t in self.trials if t.state == TrialState.COMPLETE]

        # TODO(ohta): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
        for trial in trials:
            dominated = False
            for other in trials:
                if other._dominates(trial, self.directions):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(trial)

        return pareto_front


def _log_completed_trial(trial: Trial, result: float) -> None:
    values = multi_objective.trial.MultiObjectiveTrial(trial)._values
    _logger.info(
        "Finished trial#{} with values: {} with parameters: {}.".format(
            trial.number, values, trial.params,
        )
    )
