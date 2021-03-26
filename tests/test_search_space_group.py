from optuna._search_space_group import SearchSpaceGroup
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution


def test_search_space_group() -> None:
    search_space_group = SearchSpaceGroup()

    # No search space.
    assert search_space_group.group == []

    # No distributions.
    search_space_group.add_distributions({})
    assert search_space_group.group == []

    # Add a single distribution.
    search_space_group.add_distributions({"x": IntUniformDistribution(low=0, high=10)})
    assert search_space_group.group == [{"x": IntUniformDistribution(low=0, high=10)}]

    # Add a same single distribution.
    search_space_group.add_distributions({"x": IntUniformDistribution(low=0, high=10)})
    assert search_space_group.group == [{"x": IntUniformDistribution(low=0, high=10)}]

    # Add disjoint distributions.
    search_space_group.add_distributions(
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        }
    )
    assert search_space_group.group == [
        {"x": IntUniformDistribution(low=0, high=10)},
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
    ]

    # Add distributions, which include one of search spaces in the group.
    search_space_group.add_distributions(
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
            "u": LogUniformDistribution(low=1e-2, high=1e2),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        }
    )
    assert search_space_group.group == [
        {"x": IntUniformDistribution(low=0, high=10)},
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
        {
            "u": LogUniformDistribution(low=1e-2, high=1e2),
            "v": CategoricalDistribution(choices=["A", "B", "C"]),
        },
    ]

    # Add a distribution, which is included by one of search spaces in the group.
    search_space_group.add_distributions({"u": LogUniformDistribution(low=1e-2, high=1e2)})
    assert search_space_group.group == [
        {"x": IntUniformDistribution(low=0, high=10)},
        {
            "y": IntUniformDistribution(low=0, high=10),
            "z": UniformDistribution(low=-3, high=3),
        },
        {"u": LogUniformDistribution(low=1e-2, high=1e2)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
    ]

    # Add distributions whose intersection with one of search spaces in the group is not empty.
    search_space_group.add_distributions(
        {
            "y": IntUniformDistribution(low=0, high=10),
            "w": IntLogUniformDistribution(low=2, high=8),
        }
    )
    assert search_space_group.group == [
        {"x": IntUniformDistribution(low=0, high=10)},
        {"u": LogUniformDistribution(low=1e-2, high=1e2)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"y": IntUniformDistribution(low=0, high=10)},
        {"z": UniformDistribution(low=-3, high=3)},
        {"w": IntLogUniformDistribution(low=2, high=8)},
    ]

    # Add distributions which include some of search spaces in the group.
    search_space_group.add_distributions(
        {
            "y": IntUniformDistribution(low=0, high=10),
            "w": IntLogUniformDistribution(low=2, high=8),
            "t": UniformDistribution(low=10, high=100),
        }
    )
    assert search_space_group.group == [
        {"x": IntUniformDistribution(low=0, high=10)},
        {"u": LogUniformDistribution(low=1e-2, high=1e2)},
        {"v": CategoricalDistribution(choices=["A", "B", "C"])},
        {"y": IntUniformDistribution(low=0, high=10)},
        {"z": UniformDistribution(low=-3, high=3)},
        {"w": IntLogUniformDistribution(low=2, high=8)},
        {"t": UniformDistribution(low=10, high=100)},
    ]
