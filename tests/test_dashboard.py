import tempfile

import optuna
import optuna.trial


def _create_some_study():
    # type: () -> optuna.Study

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        x = trial.suggest_uniform("x", -10, 10)
        y = trial.suggest_loguniform("y", 10, 20)
        z = trial.suggest_categorical("z", (10.0, 20.5, 30.0))
        assert isinstance(z, float)

        return x ** 2 + y ** 2 + z

    study = optuna.create_study()
    study.optimize(f, n_trials=100)
    return study


def test_write():
    # type: () -> None

    study = _create_some_study()

    with tempfile.NamedTemporaryFile("r") as tf:
        optuna.dashboard._write(study, tf.name)

        html = tf.read()
        assert "<body>" in html
        assert "bokeh" in html
