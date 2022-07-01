from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization._intermediate_values import _get_intermediate_plot
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization._intermediate_values import _IntermediatePlotInfo
from optuna.visualization._intermediate_values import _TrialInfo


def test_intermediate_plot_info() -> None:
    # Test with no trials.
    study = prepare_study_with_trials(no_trials=True)

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo([])

    # Test with a trial with intermediate values.
    def objective(trial: Trial, report_intermediate_values: bool) -> float:
        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    study = create_study()
    study.optimize(lambda t: objective(t, True), n_trials=1)

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(
        [_TrialInfo(0, [(0, 1.0), (1, 2.0)])]
    )

    # Test a study with one trial with intermediate values and
    # one trial without intermediate values.
    # Expect the trial with no intermediate values to be ignored.
    study.optimize(lambda t: objective(t, False), n_trials=1)

    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo(
        [_TrialInfo(0, [(0, 1.0), (1, 2.0)])]
    )

    # Test a study of only one trial that has no intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo([])

    # Ignore failed trials.
    def fail_objective(trial: Trial) -> float:
        trial.report(1.0, step=0)
        trial.report(2.0, step=1)
        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    assert _get_intermediate_plot_info(study) == _IntermediatePlotInfo([])


def test_plot_intermediate_values() -> None:
    figure = _get_intermediate_plot(_IntermediatePlotInfo([]))
    assert not figure.data

    # Test with a trial with intermediate values.
    figure = _get_intermediate_plot(_IntermediatePlotInfo([_TrialInfo(0, [(0, 1.0), (1, 2.0)])]))
    assert len(figure.data) == 1
    assert figure.data[0].x == (0, 1)
    assert figure.data[0].y == (1.0, 2.0)
