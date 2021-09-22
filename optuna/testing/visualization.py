from optuna import Study
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.trial import create_trial


def prepare_study_with_trials(
    no_trials: bool = False,
    less_than_two: bool = False,
    more_than_three: bool = False,
    with_c_d: bool = True,
    n_objectives: int = 1,
) -> Study:

    """Prepare a study for tests.

    Args:
        no_trials: If :obj:`False`, create a study with no trials.
        less_than_two: If :obj:`True`, create a study with two/four hyperparameters where
            'param_a' (and 'param_c') appear(s) only once while 'param_b' (and 'param_d')
            appear(s) twice in `study.trials`.
        more_than_three: If :obj:`True`, create a study with two/four hyperparameters where
            'param_a' (and 'param_c') appear(s) only three times while 'param_b' (and 'param_d')
            appear(s) four times in `study.trials`.
        with_c_d: If :obj:`True`, the study has four hyperparameters named 'param_a',
            'param_b', 'param_c', and 'param_d'. Otherwise, there are only two
            hyperparameters ('param_a' and 'param_b').
        n_objectives: Number of objective values.

    Returns:
        :class:`~optuna.study.Study`

    """

    study = create_study(directions=["minimize"] * n_objectives)
    if no_trials:
        return study
    study.add_trial(
        create_trial(
            values=[0.0] * n_objectives,
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
            values=[2.0] * n_objectives,
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
            values=[1.0] * n_objectives,
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

    if more_than_three:
        study.add_trial(
            create_trial(
                values=[1.5] * n_objectives,
                params={"param_a": 0.5, "param_b": 1.5, "param_c": 2.0, "param_d": 3.0}
                if with_c_d
                else {"param_a": 0.5, "param_b": 1.5},
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
