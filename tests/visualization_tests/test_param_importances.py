from io import BytesIO
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytest

from optuna.distributions import FloatDistribution
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance._base import BaseImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_param_importances as plotly_plot_param_importances
from optuna.visualization._param_importances import _get_importances_info
from optuna.visualization._param_importances import _ImportancesInfo
from optuna.visualization._plotly_imports import go
from optuna.visualization.matplotlib import plot_param_importances as plt_plot_param_importances
from optuna.visualization.matplotlib._matplotlib_imports import Axes
from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrize_plot_param_importances = pytest.mark.parametrize(
    "plot_param_importances", [plotly_plot_param_importances, plt_plot_param_importances]
)


def _create_study_with_failed_trial() -> Study:
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    return study


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_importances_info(
            study=study, evaluator=None, params=None, target=None, target_name="Objective Value"
        )


@parametrize_plot_param_importances
def test_plot_param_importances_customized_target_name(
    plot_param_importances: Callable[..., Any]
) -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_param_importances(study, params=params, target_name="Target Name")
    if isinstance(figure, go.Figure):
        assert figure.layout.xaxis.title.text == "Importance for Target Name"
    elif isinstance(figure, Axes):
        assert figure.figure.axes[0].get_xlabel() == "Importance for Target Name"


@parametrize_plot_param_importances
@pytest.mark.parametrize(
    "specific_create_study",
    [
        create_study,
        _create_study_with_failed_trial,
        prepare_study_with_trials,
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        [],
        ["param_a"],
        None,
    ],
)
def test_plot_param_importances(
    plot_param_importances: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    params: Optional[List[str]],
) -> None:
    study = specific_create_study()
    figure = plot_param_importances(study, params=params)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())


@pytest.mark.parametrize(
    "specific_create_study",
    [create_study, _create_study_with_failed_trial],
)
@pytest.mark.parametrize(
    "params",
    [
        [],
        ["param_a"],
        None,
    ],
)
def test_get_param_importances_info_empty(
    specific_create_study: Callable[[], Study], params: Optional[List[str]]
) -> None:
    study = specific_create_study()
    info = _get_importances_info(
        study, None, params=params, target=None, target_name="Objective Value"
    )
    assert info == _ImportancesInfo(
        importance_values=[], param_names=[], importance_labels=[], target_name="Objective Value"
    )


def test_switch_label_when_param_insignificant() -> None:
    def _objective(trial: Trial) -> int:
        x = trial.suggest_int("x", 0, 2)
        _ = trial.suggest_int("y", -1, 1)
        return x**2

    study = create_study()
    for x in range(1, 3):
        study.enqueue_trial({"x": x, "y": 0})

    study.optimize(_objective, n_trials=2)

    info = _get_importances_info(study, None, None, None, "Objective Value")

    # Test if label for `y` param has been switched to `<0.01`.
    assert info.importance_labels == ["<0.01", "1.00"]


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "evaluator",
    [MeanDecreaseImpurityImportanceEvaluator(seed=10), FanovaImportanceEvaluator(seed=10)],
)
@pytest.mark.parametrize("n_trials", [0, 10])
def test_get_info_importances_nonfinite_removed(
    inf_value: float, evaluator: BaseImportanceEvaluator, n_trials: int
) -> None:
    def _objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    seed = 13
    target_name = "Objective Value"

    study = create_study(sampler=RandomSampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials)

    # Create param importances info without inf value.
    info_without_inf = _get_importances_info(
        study, evaluator=evaluator, params=None, target=None, target_name=target_name
    )

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            value=inf_value,
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )

    # Create param importances info with inf value.
    info_with_inf = _get_importances_info(
        study, evaluator=evaluator, params=None, target=None, target_name=target_name
    )

    # Obtained info instances should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert info_with_inf == info_without_inf


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "evaluator",
    [MeanDecreaseImpurityImportanceEvaluator(seed=10), FanovaImportanceEvaluator(seed=10)],
)
@pytest.mark.parametrize("n_trial", [0, 10])
def test_multi_objective_trial_with_infinite_value_ignored(
    target_idx: int, inf_value: float, evaluator: BaseImportanceEvaluator, n_trial: int
) -> None:
    def _multi_objective_function(trial: Trial) -> Tuple[float, float]:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1, x2 * x3

    seed = 13
    target_name = "Target Name"

    study = create_study(directions=["minimize", "minimize"], sampler=RandomSampler(seed=seed))
    study.optimize(_multi_objective_function, n_trials=n_trial)

    # Create param importances info without inf value.
    info_without_inf = _get_importances_info(
        study,
        evaluator=evaluator,
        params=None,
        target=lambda t: t.values[target_idx],
        target_name=target_name,
    )

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            values=[inf_value, inf_value],
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )

    # Create param importances info with inf value.
    info_with_inf = _get_importances_info(
        study,
        evaluator=evaluator,
        params=None,
        target=lambda t: t.values[target_idx],
        target_name=target_name,
    )

    # Obtained info instances should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert info_with_inf == info_without_inf
