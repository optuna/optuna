from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.visualization.utils import _is_log_scale
from optuna.visualization.utils import is_available


def test_is_log_scale():
    # type: () -> None

    study = create_study()
    study._append_trial(
        value=0.0,
        params={"param_linear": 1.0,},
        distributions={"param_linear": UniformDistribution(0.0, 3.0),},
    )
    study._append_trial(
        value=2.0,
        params={"param_linear": 2.0, "param_log": 1e-3,},
        distributions={
            "param_linear": UniformDistribution(0.0, 3.0),
            "param_log": LogUniformDistribution(1e-5, 1.0),
        },
    )
    assert _is_log_scale(study.trials, "param_log")
    assert not _is_log_scale(study.trials, "param_linear")


def _is_plotly_available():
    # type: () -> bool

    try:
        import plotly  # NOQA

        available = True
    except Exception:
        available = False
    return available


def test_visualization_is_available():
    # type: () -> None

    assert is_available() == _is_plotly_available()
