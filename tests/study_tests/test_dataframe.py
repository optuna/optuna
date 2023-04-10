from __future__ import annotations

import pandas as pd
import pytest

from optuna import create_study
from optuna import create_trial
from optuna import Trial
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


def test_study_trials_dataframe_with_no_trials() -> None:
    study_with_no_trials = create_study()
    trials_df = study_with_no_trials.trials_dataframe()
    assert trials_df.empty


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "attrs",
    [
        (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),
        (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
            "intermediate_values",
            "_trial_id",
            "distributions",
        ),
    ],
)
@pytest.mark.parametrize("multi_index", [True, False])
def test_trials_dataframe(storage_mode: str, attrs: tuple[str, ...], multi_index: bool) -> None:
    def f(trial: Trial) -> float:
        x = trial.suggest_int("x", 1, 1)
        y = trial.suggest_categorical("y", (2.5,))
        trial.set_user_attr("train_loss", 3)
        trial.storage.set_trial_system_attr(trial._trial_id, "foo", "bar")
        value = x + y  # 3.5

        # Test reported intermediate values, although it in practice is not "intermediate".
        trial.report(value, step=0)

        return value

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=3)
        df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)
        # Change index to access rows via trial number.
        if multi_index:
            df.set_index(("number", ""), inplace=True, drop=False)
        else:
            df.set_index("number", inplace=True, drop=False)
        assert len(df) == 3

        # Number columns are as follows (total of 13):
        #   non-nested: 6 (number, value, state, datetime_start, datetime_complete, duration)
        #   params: 2
        #   distributions: 2
        #   user_attrs: 1
        #   system_attrs: 1
        #   intermediate_values: 1
        expected_n_columns = len(attrs)
        if "params" in attrs:
            expected_n_columns += 1
        if "distributions" in attrs:
            expected_n_columns += 1
        assert len(df.columns) == expected_n_columns

        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == "COMPLETE"
            assert df.value[i] == 3.5
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)

            if multi_index:
                if "distributions" in attrs:
                    assert ("distributions", "x") in df.columns
                    assert ("distributions", "y") in df.columns
                if "_trial_id" in attrs:
                    assert ("trial_id", "") in df.columns  # trial_id depends on other tests.
                if "duration" in attrs:
                    assert ("duration", "") in df.columns

                assert df.params.x[i] == 1
                assert df.params.y[i] == 2.5
                assert df.user_attrs.train_loss[i] == 3
                assert df.system_attrs.foo[i] == "bar"
            else:
                if "distributions" in attrs:
                    assert "distributions_x" in df.columns
                    assert "distributions_y" in df.columns
                if "_trial_id" in attrs:
                    assert "trial_id" in df.columns  # trial_id depends on other tests.
                if "duration" in attrs:
                    assert "duration" in df.columns

                assert df.params_x[i] == 1
                assert df.params_y[i] == 2.5
                assert df.user_attrs_train_loss[i] == 3
                assert df.system_attrs_foo[i] == "bar"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_trials_dataframe_with_failure(storage_mode: str) -> None:
    def f(trial: Trial) -> float:
        x = trial.suggest_int("x", 1, 1)
        y = trial.suggest_categorical("y", (2.5,))
        trial.set_user_attr("train_loss", 3)
        raise ValueError()
        return x + y  # 3.5

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=3, catch=(ValueError,))
        df = study.trials_dataframe()
        # Change index to access rows via trial number.
        df.set_index("number", inplace=True, drop=False)
        assert len(df) == 3
        # non-nested: 6, params: 2, user_attrs: 1 system_attrs: 0
        assert len(df.columns) == 9
        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == "FAIL"
            assert df.value[i] is None
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)
            assert isinstance(df.duration[i], pd.Timedelta)
            assert df.params_x[i] == 1
            assert df.params_y[i] == 2.5
            assert df.user_attrs_train_loss[i] == 3


@pytest.mark.parametrize("attrs", [("value",), ("values",)])
@pytest.mark.parametrize("multi_index", [True, False])
def test_trials_dataframe_with_multi_objective_optimization(
    attrs: tuple[str, ...], multi_index: bool
) -> None:
    def f(trial: Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 1, 1)
        y = trial.suggest_float("y", 2, 2)

        return x + y, x**2 + y**2  # 3, 5

    # without set_metric_names()
    study = create_study(directions=["minimize", "maximize"])
    study.optimize(f, n_trials=1)
    df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)
    if multi_index:
        assert df.get("values")[0][0] == 3
        assert df.get("values")[1][0] == 5
    else:
        assert df.values_0[0] == 3
        assert df.values_1[0] == 5

    # with set_metric_names()
    study.set_metric_names(["v0", "v1"])
    df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)
    if multi_index:
        assert df.get("values")["v0"][0] == 3
        assert df.get("values")["v1"][0] == 5
    else:
        assert df.get("values_v0")[0] == 3
        assert df.get("values_v1")[0] == 5


@pytest.mark.parametrize("attrs", [("value",), ("values",)])
@pytest.mark.parametrize("multi_index", [True, False])
def test_trials_dataframe_with_multi_objective_optimization_with_fail_and_pruned(
    attrs: tuple[str, ...], multi_index: bool
) -> None:
    study = create_study(directions=["minimize", "maximize"])
    study.add_trial(create_trial(state=TrialState.FAIL))
    study.add_trial(create_trial(state=TrialState.PRUNED))
    df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)

    # without set_metric_names()
    if multi_index:
        for i in range(2):
            assert df.get("values")[0][i] is None
            assert df.get("values")[1][i] is None
    else:
        for i in range(2):
            assert df.values_0[i] is None
            assert df.values_1[i] is None

    # with set_metric_names()
    study.set_metric_names(["v0", "v1"])
    df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)
    if multi_index:
        assert df.get("values")["v0"][0] is None
        assert df.get("values")["v1"][0] is None
    else:
        assert df.get("values_v0")[0] is None
        assert df.get("values_v1")[0] is None
