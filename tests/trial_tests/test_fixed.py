from __future__ import annotations

import pytest

from optuna.trial import FixedTrial


def test_params() -> None:
    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.params == {}

    assert trial.suggest_float("x", 0, 10) == 1
    assert trial.params == params


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("positional_args_names", [[], ["step"], ["step", "log"]])
def test_suggest_int_positional_args(positional_args_names: list[str]) -> None:
    # If log is specified as positional, step must also be provided as positional.
    params = {"x": 1}
    trial = FixedTrial(params)
    kwargs = dict(step=1, log=False)
    args = [kwargs[name] for name in positional_args_names]
    # No error should not be raised even if the coding style is old.
    trial.suggest_int("x", -1, 1, *args)


def test_number() -> None:
    params = {"x": 1}
    trial = FixedTrial(params, 2)
    assert trial.number == 2

    trial = FixedTrial(params)
    assert trial.number == 0
