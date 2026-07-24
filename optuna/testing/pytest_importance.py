from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.testing.objectives import pruned_objective
from optuna.trial import create_trial
from optuna.trial import Trial


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna import Study


def _objective(trial: Trial) -> float:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1 + x2 * x3


def _multi_objective_function(trial: Trial) -> tuple[float, float]:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1, x2 * x3


def _get_study(seed: int, n_trials: int, is_multi_obj: bool) -> Study:
    # Assumes that `seed` can be fixed to reproduce identical results.
    directions = ["minimize", "minimize"] if is_multi_obj else ["minimize"]
    study = create_study(sampler=RandomSampler(seed=seed), directions=directions)
    if is_multi_obj:
        study.optimize(_multi_objective_function, n_trials=n_trials)
    else:
        study.optimize(_objective, n_trials=n_trials)

    return study


class _BaseImportanceEvaluatorTestCase:
    @pytest.fixture
    def evaluator(self) -> Callable[..., BaseImportanceEvaluator]:
        raise NotImplementedError


class BasicImportanceEvaluatorTestCase(_BaseImportanceEvaluatorTestCase):
    def test_get_param_importances_invalid_empty_study(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        study = create_study()

        importance = get_param_importances(study, evaluator=evaluator())
        assert isinstance(importance, dict)
        assert not importance

        study.optimize(pruned_objective, n_trials=3)

        importance = get_param_importances(study, evaluator=evaluator())
        assert isinstance(importance, dict)
        assert not importance

    def test_get_param_importances_invalid_single_trial(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, 3)
            return x1**2

        study = create_study()
        study.optimize(objective, n_trials=1)

        importance = get_param_importances(study, evaluator=evaluator())
        assert importance == {"x1": 1.0}

    def test_get_param_importances_invalid_no_completed_trials_params(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, 3)
            if trial.number % 2 == 0:
                _ = trial.suggest_float("x2", 0.1, 3, log=True)
                raise optuna.TrialPruned
            return x1**2

        study = create_study()
        study.optimize(objective, n_trials=3)

        # None of the trials with `x2` are completed.
        with pytest.raises(ValueError):
            get_param_importances(study, evaluator=evaluator(), params=["x2"])
        # None of the trials with `x2` are completed. Adding "x1" should not matter.
        with pytest.raises(ValueError):
            get_param_importances(study, evaluator=evaluator(), params=["x1", "x2"])
        # None of the trials contain `x3`.
        with pytest.raises(ValueError):
            get_param_importances(study, evaluator=evaluator(), params=["x3"])

    def test_get_param_importances_empty_search_space(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        def objective(trial: Trial) -> float:
            x = trial.suggest_float("x", 0, 5)
            y = trial.suggest_float("y", 1, 1)
            return 4 * x**2 + 4 * y**2

        study = create_study()
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(study, evaluator=evaluator())

        assert len(param_importance) == 2
        assert all(param in param_importance for param in ["x", "y"])
        assert param_importance["x"] > 0.0
        assert param_importance["y"] == 0.0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
    @pytest.mark.parametrize("target_idx", [0, 1, None])
    def test_evaluator_with_infinite(
        self,
        evaluator: Callable[..., BaseImportanceEvaluator],
        inf_value: float,
        target_idx: int | None,
    ) -> None:
        evaluator_instance = evaluator()
        is_multi_obj = target_idx is not None
        study = _get_study(seed=13, n_trials=10, is_multi_obj=is_multi_obj)
        target = (lambda t: t.values[target_idx]) if is_multi_obj else None
        param_importance_without_inf = evaluator_instance.evaluate(study, target=target)

        study.add_trial(
            create_trial(
                values=[inf_value] if not is_multi_obj else [inf_value, inf_value],
                params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
                distributions={
                    "x1": FloatDistribution(low=0.1, high=3),
                    "x2": FloatDistribution(low=0.1, high=3, log=True),
                    "x3": FloatDistribution(low=2, high=4, log=True),
                },
            )
        )
        param_importance_with_inf = evaluator_instance.evaluate(study, target=target)

        assert param_importance_with_inf == param_importance_without_inf

    def test_evaluator_with_only_single_dists(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        study = create_study(sampler=RandomSampler(seed=0))
        study.optimize(lambda trial: trial.suggest_float("a", 0.0, 0.0), n_trials=3)
        param_importance = evaluator().evaluate(study)

        assert param_importance == {"a": 0.0}

    def test_importance_evaluator_with_target(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        study = create_study(sampler=RandomSampler(seed=0))
        study.optimize(_objective, n_trials=3)

        evaluator_instance = evaluator()
        param_importance = evaluator_instance.evaluate(study)
        param_importance_with_target = evaluator_instance.evaluate(
            study, target=lambda t: t.params["x3"]
        )

        assert param_importance != param_importance_with_target

    @pytest.mark.parametrize("params", [[], ["x1"], ["x1", "x3"], ["x1", "x4"]])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_get_param_importances_with_params(
        self,
        evaluator: Callable[..., BaseImportanceEvaluator],
        params: list[str],
        normalize: bool,
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, 3)
            x2 = trial.suggest_float("x2", 0.1, 3, log=True)
            x3 = trial.suggest_float("x3", 0, 3, step=1)
            if trial.number % 2 == 0:
                x4 = trial.suggest_float("x4", 0.1, 3)

            value = x1**4 + x2 + x3
            if trial.number % 2 == 0:
                value += x4
            return value

        study = create_study()
        study.optimize(objective, n_trials=10)

        param_importance = get_param_importances(
            study, evaluator=evaluator(), params=params, normalize=normalize
        )

        assert len(param_importance) == len(params)
        assert all(param in param_importance for param in params)
        assert all(isinstance(name, str) for name in param_importance)
        assert all(isinstance(importance, float) for importance in param_importance.values())
        assert all(0 <= importance < float("inf") for importance in param_importance.values())
        if normalize:
            assert len(param_importance) == 0 or np.isclose(sum(param_importance.values()), 1.0)


class ConditionalImportanceEvaluatorTestCase(_BaseImportanceEvaluatorTestCase):
    @pytest.mark.parametrize(
        "params",
        [None, [], ["c"], ["x"], ["c", "x"], ["x", "y"], ["c", "x", "y"], ["d"], ["c", "d"]],
    )
    def test_conditional_parameters(
        self,
        evaluator: Callable[..., BaseImportanceEvaluator],
        params: list[str] | None,
    ) -> None:
        study = create_study()
        dists_cx: dict[str, BaseDistribution] = {
            "c": FloatDistribution(0.0, 1.0),
            "x": FloatDistribution(-2.0, 0.0),
        }
        dists_cy: dict[str, BaseDistribution] = {
            "c": FloatDistribution(0.0, 1.0),
            "y": FloatDistribution(0.0, 2.0),
        }
        trials = [
            optuna.create_trial(params={"c": 1.0, "x": -1.0}, distributions=dists_cx, value=-1.0),
            optuna.create_trial(params={"c": 0.0, "y": 1.0}, distributions=dists_cy, value=1.0),
            optuna.create_trial(params={"c": 0.8, "x": -0.8}, distributions=dists_cx, value=-0.8),
            optuna.create_trial(params={"c": 0.2, "y": 0.2}, distributions=dists_cy, value=0.2),
            optuna.create_trial(params={"c": 0.8, "x": -0.6}, distributions=dists_cx, value=-0.6),
            optuna.create_trial(params={"c": 0.2, "y": 0.3}, distributions=dists_cy, value=0.3),
        ]
        study.add_trials(trials)

        if params and "d" in params:
            with pytest.raises(ValueError):
                evaluator().evaluate(study, params=params)
            return
        importance = evaluator().evaluate(study, params=params)
        if params == []:
            assert importance == {}
            return
        assert set(importance) == set(params or ["c", "x", "y"])
        assert not all(value == 0.0 for value in importance.values()), f"{importance=}"


class NonConditionalImportanceEvaluatorTestCase(_BaseImportanceEvaluatorTestCase):
    @pytest.mark.parametrize("normalize", [True, False])
    def test_get_param_importances_non_conditional(
        self, evaluator: Callable[..., BaseImportanceEvaluator], normalize: bool
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, 3)
            x2 = trial.suggest_float("x2", 0.1, 3, log=True)
            x3 = trial.suggest_float("x3", 0, 3, step=1)
            x4 = trial.suggest_int("x4", -3, 3)
            x5 = trial.suggest_int("x5", 1, 5, log=True)
            x6 = trial.suggest_categorical("x6", [1.0, 1.1, 1.2])
            if trial.number % 2 == 0:
                x7 = trial.suggest_float("x7", 0.1, 3)

            value = x1**4 + x2 + x3 - x4**2 - x5 + x6
            if trial.number % 2 == 0:
                value += x7
            return value

        study = create_study(sampler=samplers.RandomSampler())
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(study, evaluator=evaluator(), normalize=normalize)

        assert len(param_importance) == 6
        assert all(name in param_importance for name in ["x1", "x2", "x3", "x4", "x5", "x6"])
        previous_importance = float("inf")
        for name, importance in param_importance.items():
            assert isinstance(name, str)
            assert isinstance(importance, float)
            assert importance <= previous_importance
            previous_importance = importance
        assert all(0 <= importance < float("inf") for importance in param_importance.values())
        if normalize:
            assert np.isclose(sum(param_importance.values()), 1.0)

    def test_get_param_importances_invalid_dynamic_search_space_params(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, trial.number + 0.1)
            return x1**2

        study = create_study()
        study.optimize(objective, n_trials=3)

        with pytest.raises(ValueError):
            get_param_importances(study, evaluator=evaluator(), params=["x1"])

    @pytest.mark.parametrize("normalize", [True, False])
    def test_get_param_importances_with_target(
        self, evaluator: Callable[..., BaseImportanceEvaluator], normalize: bool
    ) -> None:
        def objective(trial: Trial) -> float:
            x1 = trial.suggest_float("x1", 0.1, 3)
            x2 = trial.suggest_float("x2", 0.1, 3, log=True)
            x3 = trial.suggest_float("x3", 0, 3, step=1)
            if trial.number % 2 == 0:
                x4 = trial.suggest_float("x4", 0.1, 3)

            value = x1**4 + x2 + x3
            if trial.number % 2 == 0:
                value += x4
            return value

        study = create_study()
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(
            study,
            evaluator=evaluator(),
            target=lambda t: t.params["x1"] + t.params["x2"],
            normalize=normalize,
        )

        assert len(param_importance) == 3
        assert all(name in param_importance for name in ["x1", "x2", "x3"])
        previous_importance = float("inf")
        for name, importance in param_importance.items():
            assert isinstance(name, str)
            assert isinstance(importance, float)
            assert importance <= previous_importance
            previous_importance = importance
        assert all(0 <= importance < float("inf") for importance in param_importance.values())
        if normalize:
            assert np.isclose(sum(param_importance.values()), 1.0)


class MultiObjectiveImportanceEvaluatorTestCase(_BaseImportanceEvaluatorTestCase):
    def test_get_param_importance_target_is_none_and_study_is_multi_obj(
        self, evaluator: Callable[..., BaseImportanceEvaluator]
    ) -> None:
        def objective(trial: Trial) -> tuple[float, float]:
            x1 = trial.suggest_float("x1", 0.1, 3)
            x2 = trial.suggest_float("x2", 0.1, 3, log=True)
            x3 = trial.suggest_float("x3", 0, 3, step=1)
            x4 = trial.suggest_int("x4", -3, 3)
            x5 = trial.suggest_int("x5", 1, 5, log=True)
            x6 = trial.suggest_categorical("x6", [1.0, 1.1, 1.2])
            if trial.number % 2 == 0:
                x7 = trial.suggest_float("x7", 0.1, 3)

            value = x1**4 + x2 + x3 - x4**2 - x5 + x6
            if trial.number % 2 == 0:
                value += x7
            return value, 0.0

        study = create_study(directions=["minimize", "minimize"])
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(study, evaluator=evaluator())
        assert isinstance(param_importance, dict)
