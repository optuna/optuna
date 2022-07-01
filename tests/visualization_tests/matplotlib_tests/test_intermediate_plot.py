from io import BytesIO

import matplotlib.pyplot as plt

from optuna.visualization.matplotlib._intermediate_values import _get_intermediate_plot
from optuna.visualization._intermediate_values import _IntermediatePlotInfo, _TrialInfo


def test_plot_intermediate_values() -> None:

    # Test with no trials.
    figure = _get_intermediate_plot(_IntermediatePlotInfo([]))
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with a trial with intermediate values.
    figure = _get_intermediate_plot(_IntermediatePlotInfo([
        _TrialInfo(0, [(0, 1.0), (1, 2.0)])
    ]))
    assert len(figure.get_lines()) == 1
    assert list(figure.get_lines()[0].get_xdata()) == [0, 1]
    assert list(figure.get_lines()[0].get_ydata()) == [1.0, 2.0]
    plt.savefig(BytesIO())
