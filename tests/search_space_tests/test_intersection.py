import pytest

from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import intersection_search_space
from optuna.search_space import IntersectionSearchSpace
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial


def test_intersection_search_space() -> None:
    search_space = IntersectionSearchSpace()
    study = create_study()

    # No trial.
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # Waiting trial.
    study.enqueue_trial(
        {"y": 0, "x": 5}, {"y": FloatDistribution(-3, 3), "x": IntDistribution(0, 10)}
    )
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # First trial.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {
        "x": IntDistribution(low=0, high=10),
        "y": FloatDistribution(low=-3, high=3),
    }
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # Returned dict is sorted by parameter names.
    assert list(search_space.calculate(study).keys()) == ["x", "y"]

    # Second trial (only 'y' parameter is suggested in this trial).
    study.optimize(lambda t: t.suggest_float("y", -3, 3), n_trials=1)
    assert search_space.calculate(study) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial: Trial, exception: Exception) -> float:
        trial.suggest_float("z", 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, TrialPruned()), n_trials=1)
    assert search_space.calculate(study) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # If two parameters have the same name but different distributions,
    # those are regarded as different parameters.
    study.optimize(lambda t: t.suggest_float("y", -1, 1), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )

    # The search space remains empty once it is empty.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == intersection_search_space(
        study.get_trials(deepcopy=False)
    )


def test_intersection_search_space_class_with_different_studies() -> None:
    search_space = IntersectionSearchSpace()

    with StorageSupplier("sqlite") as storage:
        study0 = create_study(storage=storage)
        study1 = create_study(storage=storage)

        search_space.calculate(study0)
        with pytest.raises(ValueError):
            # An `IntersectionSearchSpace` instance isn't supposed to be used for multiple studies.
            search_space.calculate(study1)
