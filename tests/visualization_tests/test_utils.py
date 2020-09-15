from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.trial import create_trial
from optuna.visualization import is_available
from optuna.visualization._utils import _is_log_scale


def test_is_log_scale() -> None:

    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_linear": 1.0},
            distributions={"param_linear": UniformDistribution(0.0, 3.0)},
        )
    )
    study.add_trial(
        create_trial(
            value=2.0,
            params={"param_linear": 2.0, "param_log": 1e-3},
            distributions={
                "param_linear": UniformDistribution(0.0, 3.0),
                "param_log": LogUniformDistribution(1e-5, 1.0),
            },
        )
    )
    assert _is_log_scale(study.trials, "param_log")
    assert not _is_log_scale(study.trials, "param_linear")


def _is_plotly_available() -> bool:

    try:
        import plotly  # NOQA

        available = True
    except Exception:
        available = False
    return available


def test_visualization_is_available() -> None:

    assert is_available() == _is_plotly_available()
