import plotly.express as px
import pytest

from optuna.testing.visualization import prepare_study_with_trials
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_same_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction)

    # This value is equivalent to `colorscale="blues"`.
    expected_colors = px.colors.sequential.Blues

    # `target` is not `None`.
    counter = plot_contour(study).data[0]
    parallel_coordinate = plot_parallel_coordinate(study).data[0]["line"]

    for color in [counter, parallel_coordinate]:

        assert expected_colors == [v[1] for v in color["colorscale"]]

        if direction == "minimize":
            assert color["reversescale"]
        else:
            assert not color["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    counter = plot_contour(study, target=lambda t: t.number).data[0]
    parallel_coordinate = plot_parallel_coordinate(study, target=lambda t: t.number).data[0][
        "line"
    ]

    for color in [parallel_coordinate, counter]:

        assert expected_colors == [v[1] for v in color["colorscale"]]
        assert color["reversescale"]

    # Since `plot_slice`'s colormap depends on only trial.number, `reversecale` is not in the plot.
    color = plot_slice(study).data[0]["marker"]
    assert expected_colors == [v[1] for v in color["colorscale"]]
    assert "reversecale" not in color
