import pytest

from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers._search_space import _GroupDecomposedSearchSpace
from optuna.testing.storage import StorageSupplier
from optuna.trial import Trial


def test_group_decomposed_search_space() -> None:
    search_space = _GroupDecomposedSearchSpace()
    study = create_study()

    # No trial.
    assert search_space.calculate(study).group == []

    # A single parameter.
    study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study).group == [{"x": IntUniformDistribution(low=0, high=10)}]

    # Disjoint parameters.
    study.optimize(lambda t: t.suggest_int("y", 0, 10) + t.suggest_float("z", -3, 3), n_trials=1)
    assert search_space.calculate(study).group == [
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
        {"x": IntUniformDistribution(low=0, high=10)},
    ]

    # Parameters which include one of search spaces in the group.
    study.optimize(
        lambda t: t.suggest_int("y", 0, 10)
        + t.suggest_float("z", -3, 3)
        + t.suggest_float("u", 1e-2, 1e2, log=True)
        + bool(t.suggest_categorical("v", ["A", "B", "C"])),
        n_trials=1,
    )
    assert search_space.calculate(study).group == [
        {
            "u": LogUniformDistribution(low=1e-2, high=1e2),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        },
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
        {"x": IntUniformDistribution(low=0, high=10)},
    ]

    # A parameter which is included by one of search spaces in thew group.
    study.optimize(lambda t: t.suggest_float("u", 1e-2, 1e2, log=True), n_trials=1)
    assert search_space.calculate(study).group == [
        {"u": LogUniformDistribution(low=1e-2, high=1e2)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
        {"x": IntUniformDistribution(low=0, high=10)},
    ]

    # Parameters whose intersection with one of search spaces in the group is not empty.
    study.optimize(
        lambda t: t.suggest_int("y", 0, 10) + t.suggest_int("w", 2, 8, log=True), n_trials=1
    )
    assert search_space.calculate(study).group == [
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"y": IntUniformDistribution(low=0, high=10)},
        {"w": IntLogUniformDistribution(low=2, high=8)},
        {"u": LogUniformDistribution(low=1e-2, high=1e2)},
        {"z": UniformDistribution(low=-3, high=3)},
        {"x": IntUniformDistribution(low=0, high=10)},
    ]

    search_space = _GroupDecomposedSearchSpace()
    study = create_study()

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial: Trial, exception: Exception) -> float:

        trial.suggest_float("a", 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, TrialPruned()), n_trials=1)
    assert search_space.calculate(study).group == []

    # If two parameters have the same name but different distributions,
    # the second one takes priority.
    study.optimize(lambda t: t.suggest_float("a", -1, 1), n_trials=1)
    study.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=1)
    assert search_space.calculate(study).group == [{"a": UniformDistribution(low=0, high=1)}]


def test_group_decomposed_search_space_with_different_studies() -> None:
    search_space = _GroupDecomposedSearchSpace()

    with StorageSupplier("sqlite") as storage:
        study0 = create_study(storage=storage)
        study1 = create_study(storage=storage)

        search_space.calculate(study0)
        with pytest.raises(ValueError):
            # `_GroupDecomposedSearchSpace` isn't supposed to be used for multiple studies.
            search_space.calculate(study1)
