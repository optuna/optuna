import math
from typing import Dict

import numpy as np
import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization._utils import COLOR_SCALE


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_parallel_coordinate(study)


def test_plot_parallel_coordinate() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (1.0, 2.0)
    assert figure.data[0]["dimensions"][2]["values"] == (2.0, 1.0)

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    assert len(figure.data[0]["dimensions"]) == 2
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    assert len(figure.data[0]["dimensions"]) == 2
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (1.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (2.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Target Name"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (1.0, 2.0)
    assert figure.data[0]["dimensions"][2]["values"] == (2.0, 1.0)

    # Test with wrong params that do not exist in trials
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        plot_parallel_coordinate(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0


def test_plot_parallel_coordinate_categorical_params() -> None:
    # Test with categorical params that cannot be converted to numeral.
    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "category_b": CategoricalDistribution(("net", "una")),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "category_b": "net"},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "opt", "category_b": "una"},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("preferred", "opt")
    assert figure.data[0]["dimensions"][2]["label"] == "category_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["ticktext"] == ("net", "una")


def test_plot_parallel_coordinate_categorical_numeric_params() -> None:
    # Test with categorical params that can be interpreted as numeric params.
    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution((1, 2)),
        "category_b": CategoricalDistribution((10, 20, 30)),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": 2, "category_b": 20},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=1.0,
            params={"category_a": 1, "category_b": 30},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": 2, "category_b": 10},
            distributions=distributions,
        )
    )

    # Trials are sorted by using param_a and param_b, i.e., trial#1, trial#2, and trial#0.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (1.0, 2.0, 0.0)
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["values"] == (0, 1, 1)
    assert figure.data[0]["dimensions"][1]["ticktext"] == (1, 2)
    assert figure.data[0]["dimensions"][1]["tickvals"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["label"] == "category_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0, 2)
    assert figure.data[0]["dimensions"][2]["values"] == (2, 0, 1)
    assert figure.data[0]["dimensions"][2]["ticktext"] == (10, 20, 30)
    assert figure.data[0]["dimensions"][2]["tickvals"] == (0, 1, 2)


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params.
    study_log_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_log_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": 10},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2e-5, "param_b": 200},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=0.1,
            params={"param_a": 1e-4, "param_b": 30},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study_log_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 1.0, 0.1)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (-6.0, -4.0)
    assert figure.data[0]["dimensions"][1]["values"] == (-6, math.log10(2e-5), -4)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("1e-06", "1e-05", "0.0001")
    assert figure.data[0]["dimensions"][1]["tickvals"] == (-6, -5, -4.0)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (1.0, math.log10(200))
    assert figure.data[0]["dimensions"][2]["values"] == (1.0, math.log10(200), math.log10(30))
    assert figure.data[0]["dimensions"][2]["ticktext"] == ("10", "100", "200")
    assert figure.data[0]["dimensions"][2]["tickvals"] == (1.0, 2.0, math.log10(200))


def test_plot_parallel_coordinate_unique_hyper_param() -> None:
    # Test case when one unique value is suggested during the optimization.

    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "param_b": 30},
            distributions=distributions,
        )
    )

    # Both hyperparameters contain unique values.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 0.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0,)
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 0)
    assert figure.data[0]["dimensions"][1]["values"] == (0.0,)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("preferred",)
    assert figure.data[0]["dimensions"][1]["tickvals"] == (0,)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (math.log10(30), math.log10(30))
    assert figure.data[0]["dimensions"][2]["values"] == (math.log10(30),)
    assert figure.data[0]["dimensions"][2]["ticktext"] == ("30",)
    assert figure.data[0]["dimensions"][2]["tickvals"] == (math.log10(30),)

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "preferred", "param_b": 20},
            distributions=distributions,
        )
    )

    # Still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 0)
    assert figure.data[0]["dimensions"][1]["values"] == (0.0, 0.0)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("preferred",)
    assert figure.data[0]["dimensions"][1]["tickvals"] == (0,)


def test_plot_parallel_coordinate_with_categorical_numeric_params() -> None:
    # Test with sample from multiple distributions including categorical params
    # that can be interpreted as numeric params.
    study_multi_distro_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "param_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": CategoricalDistribution((1, 2, 10)),
        "param_c": FloatDistribution(1, 1000, log=True),
        "param_d": CategoricalDistribution((1, -1, 2)),
    }
    study_multi_distro_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": "preferred", "param_b": 2, "param_c": 30, "param_d": 2},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": "opt", "param_b": 1, "param_c": 200, "param_d": 2},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=2.0,
            params={"param_a": "preferred", "param_b": 10, "param_c": 10, "param_d": 1},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=3.0,
            params={"param_a": "opt", "param_b": 2, "param_c": 10, "param_d": -1},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study_multi_distro_params)
    assert len(figure.data[0]["dimensions"]) == 5
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 3.0)
    assert figure.data[0]["dimensions"][0]["values"] == (1.0, 3.0, 0.0, 2.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["values"] == (1, 1, 0, 0)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("preferred", "opt")
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0, 2)
    assert figure.data[0]["dimensions"][2]["values"] == (0, 1, 1, 2)
    assert figure.data[0]["dimensions"][2]["ticktext"] == (1, 2, 10)
    assert figure.data[0]["dimensions"][3]["label"] == "param_c"
    assert figure.data[0]["dimensions"][3]["range"] == (1.0, math.log10(200))
    assert figure.data[0]["dimensions"][3]["values"] == (math.log10(200), 1.0, math.log10(30), 1.0)
    assert figure.data[0]["dimensions"][3]["ticktext"] == ("10", "100", "200")
    assert figure.data[0]["dimensions"][3]["tickvals"] == (1.0, 2.0, math.log10(200))
    assert figure.data[0]["dimensions"][4]["label"] == "param_d"
    assert figure.data[0]["dimensions"][4]["range"] == (0, 2)
    assert figure.data[0]["dimensions"][4]["values"] == (2, 0, 2, 1)
    assert figure.data[0]["dimensions"][4]["ticktext"] == (-1, 1, 2)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction)

    # `target` is `None`.
    line = plot_parallel_coordinate(study).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    if direction == "minimize":
        assert line["reversescale"]
    else:
        assert not line["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    line = plot_parallel_coordinate(study, target=lambda t: t.number).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    assert line["reversescale"]

    # Multi-objective optimization.
    study = prepare_study_with_trials(with_c_d=False, n_objectives=2, direction=direction)
    line = plot_parallel_coordinate(study, target=lambda t: t.number).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    assert line["reversescale"]


def test_plot_parallel_coordinate_only_missing_params() -> None:
    # When all trials contain only a part of parameters,
    # the plot returns an empty figure.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_b": 200},
            distributions={
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )

    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_removed(value: float) -> None:

    study = prepare_study_with_trials(value_for_first_trial=value)
    figure = plot_parallel_coordinate(study)
    assert all(np.isfinite(figure.data[0]["dimensions"][0]["values"]))


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf"), float("nan")))
def test_nonfinite_multiobjective(objective: int, value: float) -> None:

    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    figure = plot_parallel_coordinate(study, target=lambda t: t.values[objective])
    assert all(np.isfinite(figure.data[0]["dimensions"][0]["values"]))
