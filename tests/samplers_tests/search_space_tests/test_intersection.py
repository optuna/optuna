from collections import OrderedDict

import pytest

from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import FloatDistribution
from optuna.distributions import IntUniformDistribution
from optuna.samplers import intersection_search_space
from optuna.samplers import IntersectionSearchSpace
from optuna.testing.storage import StorageSupplier
from optuna.trial import Trial


def test_intersection_search_space() -> None:
    search_space = IntersectionSearchSpace()
    study = create_study()

    # No trial.
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(study)

    # First trial.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {
        "x": IntUniformDistribution(low=0, high=10),
        "y": FloatDistribution(low=-3, high=3),
    }
    assert search_space.calculate(study) == intersection_search_space(study)

    # Returning sorted `OrderedDict` instead of `dict`.
    assert search_space.calculate(study, ordered_dict=True) == OrderedDict(
        [
            ("x", IntUniformDistribution(low=0, high=10)),
            ("y", FloatDistribution(low=-3, high=3)),
        ]
    )
    assert search_space.calculate(study, ordered_dict=True) == intersection_search_space(
        study, ordered_dict=True
    )

    # Second trial (only 'y' parameter is suggested in this trial).
    study.optimize(lambda t: t.suggest_float("y", -3, 3), n_trials=1)
    assert search_space.calculate(study) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == intersection_search_space(study)

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial: Trial, exception: Exception) -> float:

        trial.suggest_float("z", 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, TrialPruned()), n_trials=1)
    assert search_space.calculate(study) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == intersection_search_space(study)

    # If two parameters have the same name but different distributions,
    # those are regarded as different parameters.
    study.optimize(lambda t: t.suggest_float("y", -1, 1), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(study)

    # The search space remains empty once it is empty.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(study)


def test_intersection_search_space_class_with_different_studies() -> None:
    search_space = IntersectionSearchSpace()

    with StorageSupplier("sqlite") as storage:
        study0 = create_study(storage=storage)
        study1 = create_study(storage=storage)

        search_space.calculate(study0)
        with pytest.raises(ValueError):
            # An `IntersectionSearchSpace` instance isn't supposed to be used for multiple studies.
            search_space.calculate(study1)
