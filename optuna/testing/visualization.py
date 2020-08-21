from optuna.distributions import UniformDistribution
from optuna import Study
from optuna.study import create_study
from optuna.trial import create_trial


def prepare_study_with_trials(
    no_trials: bool = False, less_than_two: bool = False, with_c_d: bool = True
) -> Study:

    """Prepare a study for tests.

    Args:
        no_trials: If ``False``, create a study with no trials.
        less_than_two: If ``True``, create a study with two/four hyperparameters where
            'param_a' (and 'param_c') appear(s) only once while 'param_b' (and 'param_b')
            appear(s) twice in `study.trials`.
        with_c_d: If ``True``, the study has four hyperparameters named 'param_a',
            'param_b', 'param_c', and 'param_d'. Otherwise, there are only two
            hyperparameters ('param_a' and 'param_b').

    Returns:
        :class:`~optuna.study.Study`

    """

    study = create_study()
    if no_trials:
        return study
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1.0, "param_b": 2.0, "param_c": 3.0, "param_d": 4.0}
            if with_c_d
            else {"param_a": 1.0, "param_b": 2.0},
            distributions={
                "param_a": UniformDistribution(0.0, 3.0),
                "param_b": UniformDistribution(0.0, 3.0),
                "param_c": UniformDistribution(2.0, 5.0),
                "param_d": UniformDistribution(2.0, 5.0),
            }
            if with_c_d
            else {
                "param_a": UniformDistribution(0.0, 3.0),
                "param_b": UniformDistribution(0.0, 3.0),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=2.0,
            params={"param_b": 0.0, "param_d": 4.0} if with_c_d else {"param_b": 0.0},
            distributions={
                "param_b": UniformDistribution(0.0, 3.0),
                "param_d": UniformDistribution(2.0, 5.0),
            }
            if with_c_d
            else {"param_b": UniformDistribution(0.0, 3.0)},
        )
    )
    if less_than_two:
        return study

    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2.5, "param_b": 1.0, "param_c": 4.5, "param_d": 2.0}
            if with_c_d
            else {"param_a": 2.5, "param_b": 1.0},
            distributions={
                "param_a": UniformDistribution(0.0, 3.0),
                "param_b": UniformDistribution(0.0, 3.0),
                "param_c": UniformDistribution(2.0, 5.0),
                "param_d": UniformDistribution(2.0, 5.0),
            }
            if with_c_d
            else {
                "param_a": UniformDistribution(0.0, 3.0),
                "param_b": UniformDistribution(0.0, 3.0),
            },
        )
    )
    return study
