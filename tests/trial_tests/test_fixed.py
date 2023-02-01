from optuna.trial import FixedTrial


def test_params() -> None:
    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.params == {}

    assert trial.suggest_float("x", 0, 10) == 1
    assert trial.params == params


def test_number() -> None:
    params = {"x": 1}
    trial = FixedTrial(params, 2)
    assert trial.number == 2

    trial = FixedTrial(params)
    assert trial.number == 0
