import datetime
from typing import cast
from typing import List
from typing import Tuple

import pytest

import optuna
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.study import StudyDirection
from optuna.trial import TrialState


def test_suggest() -> None:
    study = optuna.multi_objective.create_study(["maximize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> Tuple[float, float]:
        p0 = trial.suggest_float("p0", -10, 10)
        p1 = trial.suggest_uniform("p1", 3, 5)
        p2 = trial.suggest_loguniform("p2", 0.00001, 0.1)
        p3 = trial.suggest_discrete_uniform("p3", 100, 200, q=5)
        p4 = trial.suggest_int("p4", -20, -15)
        p5 = cast(int, trial.suggest_categorical("p5", [7, 1, 100]))
        p6 = trial.suggest_float("p6", -10, 10, step=1.0)
        p7 = trial.suggest_int("p7", 1, 7, log=True)
        return (
            p0 + p1 + p2,
            p3 + p4 + p5 + p6 + p7,
        )

    study.optimize(objective, n_trials=10)


def test_report() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(
        trial: optuna.multi_objective.trial.MultiObjectiveTrial,
    ) -> Tuple[float, float, float]:
        if trial.number == 0:
            trial.report((1, 2, 3), 1)
            trial.report((10, 20, 30), 2)
        return 100, 200, 300

    study.optimize(objective, n_trials=2)

    trial = study.trials[0]
    assert trial.intermediate_values == {1: (1, 2, 3), 2: (10, 20, 30)}
    assert trial.values == (100, 200, 300)
    assert trial.last_step == 2

    trial = study.trials[1]
    assert trial.intermediate_values == {}
    assert trial.values == (100, 200, 300)
    assert trial.last_step is None


def test_number() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(
        trial: optuna.multi_objective.trial.MultiObjectiveTrial, number: int
    ) -> List[float]:
        assert trial.number == number
        return [0, 0, 0]

    for i in range(10):
        study.optimize(lambda t: objective(t, i), n_trials=1)

    for i, trial in enumerate(study.trials):
        assert trial.number == i


def test_user_attrs() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        trial.set_user_attr("foo", "bar")
        assert trial.user_attrs == {"foo": "bar"}

        trial.set_user_attr("baz", "qux")
        assert trial.user_attrs == {"foo": "bar", "baz": "qux"}

        trial.set_user_attr("foo", "quux")
        assert trial.user_attrs == {"foo": "quux", "baz": "qux"}

        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert study.trials[0].user_attrs == {"foo": "quux", "baz": "qux"}


def test_system_attrs() -> None:
    # We use `RandomMultiObjectiveSampler` here because the default `NSGAIIMultiObjectiveSampler`
    # sets its own system attributes when sampling (these attributes would become noise in this
    # test case).
    sampler = optuna.multi_objective.samplers.RandomMultiObjectiveSampler()
    study = optuna.multi_objective.create_study(
        ["maximize", "minimize", "maximize"], sampler=sampler
    )

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        trial.set_system_attr("foo", "bar")
        assert trial.system_attrs == {"foo": "bar"}
        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert study.trials[0].system_attrs == {"foo": "bar"}


def test_params_and_distributions() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        x = trial.suggest_uniform("x", 0, 10)

        assert set(trial.params.keys()) == {"x"}
        assert set(trial.distributions.keys()) == {"x"}
        assert isinstance(trial.distributions["x"], optuna.distributions.UniformDistribution)

        return [x, x, x]

    study.optimize(objective, n_trials=1)

    trial = study.trials[0]
    assert set(trial.params.keys()) == {"x"}
    assert set(trial.distributions.keys()) == {"x"}
    assert isinstance(trial.distributions["x"], optuna.distributions.UniformDistribution)


def test_datetime() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        assert isinstance(trial.datetime_start, datetime.datetime)

        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert isinstance(study.trials[0].datetime_start, datetime.datetime)
    assert isinstance(study.trials[0].datetime_complete, datetime.datetime)


def test_dominates() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    def create_trial(
        values: List[float], state: TrialState = TrialState.COMPLETE
    ) -> FrozenMultiObjectiveTrial:
        n_objectives = len(values)
        trial = optuna.trial.FrozenTrial(
            state=state,
            intermediate_values={i: v for i, v in enumerate(values)},
            # The following attributes aren't used in this test case.
            number=0,
            value=None,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            trial_id=0,
        )
        return FrozenMultiObjectiveTrial(n_objectives, trial)

    # The numbers of objectives for `t0` and `t1` don't match.
    with pytest.raises(ValueError):
        t0 = create_trial([1])  # One objective.
        t1 = create_trial([1, 2])  # Two objectives.
        t0._dominates(t1, directions)

    # The numbers of objectives and directions don't match.
    with pytest.raises(ValueError):
        t0 = create_trial([1])  # One objective.
        t1 = create_trial([1])  # One objective.
        t0._dominates(t1, directions)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 2])
    t1 = create_trial([1, 1])
    assert t0._dominates(t1, directions)
    assert not t1._dominates(t0, directions)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 1])
    assert t0._dominates(t1, directions)
    assert not t1._dominates(t0, directions)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([1, 1])
    t1 = create_trial([1, 1])
    assert not t0._dominates(t1, directions)
    assert not t1._dominates(t0, directions)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 2])
    assert not t0._dominates(t1, directions)
    assert not t1._dominates(t0, directions)

    for t0_state in [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED]:
        t0 = create_trial([1, 1], t0_state)

        for t1_state in [
            TrialState.COMPLETE,
            TrialState.FAIL,
            TrialState.WAITING,
            TrialState.PRUNED,
        ]:
            # If `t0` has not the COMPLETE state, it never dominates other trials.
            t1 = create_trial([0, 2], t1_state)
            assert not t0._dominates(t1, directions)

            if t1_state == TrialState.COMPLETE:
                # If `t0` isn't COMPLETE and `t1` is COMPLETE, `t1` dominates `t0`.
                assert t1._dominates(t0, directions)
            else:
                # If `t1` isn't COMPLETE, it doesn't dominate others.
                assert not t1._dominates(t0, directions)
