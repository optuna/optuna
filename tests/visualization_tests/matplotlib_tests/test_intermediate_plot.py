from io import BytesIO

import matplotlib.pyplot as plt

from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_intermediate_values


def test_plot_intermediate_values() -> None:

    # Test with no trials.
    study = prepare_study_with_trials(no_trials=True)
    figure = plot_intermediate_values(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    def objective(trial: Trial, report_intermediate_values: bool) -> float:

        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    # Test with a trial with intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, True), n_trials=1)
    figure = plot_intermediate_values(study)
    assert len(figure.get_lines()) == 1
    assert list(figure.get_lines()[0].get_xdata()) == [0, 1]
    assert list(figure.get_lines()[0].get_ydata()) == [1.0, 2.0]
    plt.savefig(BytesIO())

    # Test a study with one trial with intermediate values and
    # one trial without intermediate values.
    # Expect the trial with no intermediate values to be ignored.
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert len(study.trials) == 2
    figure = plot_intermediate_values(study)
    assert len(figure.get_lines()) == 1
    assert list(figure.get_lines()[0].get_xdata()) == [0, 1]
    assert list(figure.get_lines()[0].get_ydata()) == [1.0, 2.0]
    plt.savefig(BytesIO())

    # Test a study of only one trial that has no intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    figure = plot_intermediate_values(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_intermediate_values(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())
