import pytest

from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import _GroupDecomposedSearchSpace
from optuna.search_space import _SearchSpaceGroup
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial


def test_search_space_group() -> None:
    search_space_group = _SearchSpaceGroup()

    # No search space.
    assert search_space_group.search_spaces == []

    # No distributions.
    search_space_group.add_distributions({})
    assert search_space_group.search_spaces == []

    # Add a single distribution.
    search_space_group.add_distributions({"x": IntDistribution(low=0, high=10)})
    assert search_space_group.search_spaces == [{"x": IntDistribution(low=0, high=10)}]

    # Add a same single distribution.
    search_space_group.add_distributions({"x": IntDistribution(low=0, high=10)})
    assert search_space_group.search_spaces == [{"x": IntDistribution(low=0, high=10)}]

    # Add disjoint distributions.
    search_space_group.add_distributions(
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        }
    )
    assert search_space_group.search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        },
    ]

    # Add distributions, which include one of search spaces in the group.
    search_space_group.add_distributions(
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
            "u": FloatDistribution(low=1e-2, high=1e2, log=True),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        }
    )
    assert search_space_group.search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        },
        {
            "u": FloatDistribution(low=1e-2, high=1e2, log=True),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        },
    ]

    # Add a distribution, which is included by one of search spaces in the group.
    search_space_group.add_distributions({"u": FloatDistribution(low=1e-2, high=1e2, log=True)})
    assert search_space_group.search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        },
        {"u": FloatDistribution(low=1e-2, high=1e2, log=True)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
    ]

    # Add distributions whose intersection with one of search spaces in the group is not empty.
    search_space_group.add_distributions(
        {
            "y": IntDistribution(low=0, high=10),
            "w": IntDistribution(low=2, high=8, log=True),
        }
    )
    assert search_space_group.search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {"y": IntDistribution(low=0, high=10)},
        {"z": FloatDistribution(low=-3, high=3)},
        {"u": FloatDistribution(low=1e-2, high=1e2, log=True)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"w": IntDistribution(low=2, high=8, log=True)},
    ]

    # Add distributions which include some of search spaces in the group.
    search_space_group.add_distributions(
        {
            "y": IntDistribution(low=0, high=10),
            "w": IntDistribution(low=2, high=8, log=True),
            "t": FloatDistribution(low=10, high=100),
        }
    )
    assert search_space_group.search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {"y": IntDistribution(low=0, high=10)},
        {"z": FloatDistribution(low=-3, high=3)},
        {"u": FloatDistribution(low=1e-2, high=1e2, log=True)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"w": IntDistribution(low=2, high=8, log=True)},
        {"t": FloatDistribution(low=10, high=100)},
    ]


def test_group_decomposed_search_space() -> None:
    search_space = _GroupDecomposedSearchSpace()
    study = create_study()

    # No trial.
    assert search_space.calculate(study).search_spaces == []

    # A single parameter.
    study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study).search_spaces == [{"x": IntDistribution(low=0, high=10)}]

    # Disjoint parameters.
    study.optimize(lambda t: t.suggest_int("y", 0, 10) + t.suggest_float("z", -3, 3), n_trials=1)
    assert search_space.calculate(study).search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        },
    ]

    # Parameters which include one of search spaces in the group.
    study.optimize(
        lambda t: t.suggest_int("y", 0, 10)
        + t.suggest_float("z", -3, 3)
        + t.suggest_float("u", 1e-2, 1e2, log=True)
        + bool(t.suggest_categorical("v", ["A", "B", "C"])),
        n_trials=1,
    )
    assert search_space.calculate(study).search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "z": FloatDistribution(low=-3, high=3),
            "y": IntDistribution(low=0, high=10),
        },
        {
            "u": FloatDistribution(low=1e-2, high=1e2, log=True),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        },
    ]

    # A parameter which is included by one of search spaces in thew group.
    study.optimize(lambda t: t.suggest_float("u", 1e-2, 1e2, log=True), n_trials=1)
    assert search_space.calculate(study).search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {
            "y": IntDistribution(low=0, high=10),
            "z": FloatDistribution(low=-3, high=3),
        },
        {"u": FloatDistribution(low=1e-2, high=1e2, log=True)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
    ]

    # Parameters whose intersection with one of search spaces in the group is not empty.
    study.optimize(
        lambda t: t.suggest_int("y", 0, 10) + t.suggest_int("w", 2, 8, log=True), n_trials=1
    )
    assert search_space.calculate(study).search_spaces == [
        {"x": IntDistribution(low=0, high=10)},
        {"y": IntDistribution(low=0, high=10)},
        {"z": FloatDistribution(low=-3, high=3)},
        {"u": FloatDistribution(low=1e-2, high=1e2, log=True)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"w": IntDistribution(low=2, high=8, log=True)},
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
    assert search_space.calculate(study).search_spaces == []

    # If two parameters have the same name but different distributions,
    # the first one takes priority.
    study.optimize(lambda t: t.suggest_float("a", -1, 1), n_trials=1)
    study.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=1)
    assert search_space.calculate(study).search_spaces == [
        {"a": FloatDistribution(low=-1, high=1)}
    ]


def test_group_decomposed_search_space_with_different_studies() -> None:
    search_space = _GroupDecomposedSearchSpace()

    with StorageSupplier("sqlite") as storage:
        study0 = create_study(storage=storage)
        study1 = create_study(storage=storage)

        search_space.calculate(study0)
        with pytest.raises(ValueError):
            # `_GroupDecomposedSearchSpace` isn't supposed to be used for multiple studies.
            search_space.calculate(study1)
