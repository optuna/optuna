import math
from typing import Dict
from typing import List
from typing import Tuple
from unittest.mock import Mock

import numpy
import pytest

from optuna.importance._fanova._tree import _FanovaTree


@pytest.fixture
def tree() -> _FanovaTree:
    sklearn_tree = Mock()
    sklearn_tree.n_features = 3
    sklearn_tree.node_count = 5
    sklearn_tree.feature = [1, 2, -1, -1, -1]
    sklearn_tree.children_left = [1, 2, -1, -1, -1]
    sklearn_tree.children_right = [4, 3, -1, -1, -1]
    sklearn_tree.value = [-1.0, -1.0, 0.1, 0.2, 0.5]
    sklearn_tree.threshold = [0.5, 1.5, -1.0, -1.0, -1.0]

    search_spaces = numpy.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]])

    return _FanovaTree(tree=sklearn_tree, search_spaces=search_spaces)


@pytest.fixture
def expected_tree_statistics() -> List[Dict[str, List]]:
    # Statistics the each node in the tree.
    return [
        {"values": [0.1, 0.2, 0.5], "weights": [0.75, 0.25, 1.0]},
        {"values": [0.1, 0.2], "weights": [0.75, 0.25]},
        {"values": [0.1], "weights": [0.75]},
        {"values": [0.2], "weights": [0.25]},
        {"values": [0.5], "weights": [1.0]},
    ]


def test_tree_variance(tree: _FanovaTree, expected_tree_statistics: List[Dict[str, List]]) -> None:
    # The root node at node index `0` holds the values and weights for all nodes in the tree.
    expected_statistics = expected_tree_statistics[0]
    expected_values = expected_statistics["values"]
    expected_weights = expected_statistics["weights"]
    expected_average_value = numpy.average(expected_values, weights=expected_weights)
    expected_variance = numpy.average(
        (expected_values - expected_average_value) ** 2, weights=expected_weights
    )

    assert math.isclose(tree.variance, expected_variance)


Size = float
NodeIndex = int
Cardinality = float


@pytest.mark.parametrize(
    "features,expected",
    [
        ([0], [([1.0], [(0, 1.0)])]),
        ([1], [([0.5], [(1, 0.5)]), ([0.5], [(4, 0.5)])]),
        ([2], [([1.5], [(2, 1.5), (4, 2.0)]), ([0.5], [(3, 0.5), (4, 2.0)])]),
        ([0, 1], [([1.0, 0.5], [(1, 0.5)]), ([1.0, 0.5], [(4, 0.5)])]),
        ([0, 2], [([1.0, 1.5], [(2, 1.5), (4, 2.0)]), ([1.0, 0.5], [(3, 0.5), (4, 2.0)])]),
        (
            [1, 2],
            [
                ([0.5, 1.5], [(2, 0.5 * 1.5)]),
                ([0.5, 1.5], [(4, 0.5 * 2.0)]),
                ([0.5, 0.5], [(3, 0.5 * 0.5)]),
                ([0.5, 0.5], [(4, 0.5 * 2.0)]),
            ],
        ),
        (
            [0, 1, 2],
            [
                ([1.0, 0.5, 1.5], [(2, 1.0 * 0.5 * 1.5)]),
                ([1.0, 0.5, 1.5], [(4, 1.0 * 0.5 * 2.0)]),
                ([1.0, 0.5, 0.5], [(3, 1.0 * 0.5 * 0.5)]),
                ([1.0, 0.5, 0.5], [(4, 1.0 * 0.5 * 2.0)]),
            ],
        ),
    ],
)
def test_tree_get_marginal_variance(
    tree: _FanovaTree,
    features: List[int],
    expected: List[Tuple[List[Size], List[Tuple[NodeIndex, Cardinality]]]],
    expected_tree_statistics: List[Dict[str, List]],
) -> None:
    features = numpy.array(features)
    variance = tree.get_marginal_variance(features)

    expected_values = []
    expected_weights = []

    for sizes, node_indices_and_corrections in expected:
        expected_split_values = []
        expected_split_weights = []

        for node_index, cardinality in node_indices_and_corrections:
            expected_statistics = expected_tree_statistics[node_index]

            expected_split_values.append(expected_statistics["values"])
            expected_split_weights.append(
                [w / cardinality for w in expected_statistics["weights"]]
            )

        expected_value = numpy.average(expected_split_values, weights=expected_split_weights)
        expected_weight = numpy.prod(numpy.array(sizes) * numpy.sum(expected_split_weights))
        expected_values.append(expected_value)
        expected_weights.append(expected_weight)

    expected_average_value = numpy.average(expected_values, weights=expected_weights)
    expected_variance = numpy.average(
        (expected_values - expected_average_value) ** 2, weights=expected_weights
    )

    assert math.isclose(variance, expected_variance)


@pytest.mark.parametrize(
    "feature_vector,expected",
    [
        ([0.5, float("nan"), float("nan")], [(0, 1.0)]),
        ([float("nan"), 0.25, float("nan")], [(1, 0.5)]),
        ([float("nan"), 0.75, float("nan")], [(4, 0.5)]),
        ([float("nan"), float("nan"), 0.75], [(2, 1.5), (4, 2.0)]),
        ([float("nan"), float("nan"), 1.75], [(3, 0.5), (4, 2.0)]),
        ([0.5, 0.25, float("nan")], [(1, 1.0 * 0.5)]),
        ([0.5, 0.75, float("nan")], [(4, 1.0 * 0.5)]),
        ([0.5, float("nan"), 0.75], [(2, 1.0 * 1.5), (4, 1.0 * 2.0)]),
        ([0.5, float("nan"), 1.75], [(3, 1.0 * 0.5), (4, 1.0 * 2.0)]),
        ([float("nan"), 0.25, 0.75], [(2, 0.5 * 1.5)]),
        ([float("nan"), 0.25, 1.75], [(3, 0.5 * 0.5)]),
        ([float("nan"), 0.75, 0.75], [(4, 0.5 * 2.0)]),
        ([float("nan"), 0.75, 1.75], [(4, 0.5 * 2.0)]),
        ([0.5, 0.25, 0.75], [(2, 1.0 * 0.5 * 1.5)]),
        ([0.5, 0.25, 1.75], [(3, 1.0 * 0.5 * 0.5)]),
        ([0.5, 0.75, 0.75], [(4, 1.0 * 0.5 * 2.0)]),
        ([0.5, 0.75, 1.75], [(4, 1.0 * 0.5 * 2.0)]),
    ],
)
def test_tree_get_marginalized_statistics(
    tree: _FanovaTree,
    feature_vector: List[float],
    expected: List[Tuple[NodeIndex, Cardinality]],
    expected_tree_statistics: List[Dict[str, List]],
) -> None:
    value, weight = tree._get_marginalized_statistics(numpy.array(feature_vector))

    expected_values = []
    expected_weights = []

    for node_index, cardinality in expected:
        expected_statistics = expected_tree_statistics[node_index]
        expected_values.append(expected_statistics["values"])
        expected_weights.append([w / cardinality for w in expected_statistics["weights"]])

    expected_value = numpy.average(expected_values, weights=expected_weights)
    expected_weight = numpy.sum(expected_weights)

    assert math.isclose(value, expected_value)
    assert math.isclose(weight, expected_weight)


def test_tree_statistics(
    tree: _FanovaTree, expected_tree_statistics: List[Dict[str, List]]
) -> None:
    statistics = tree._statistics

    for statistic, expected_statistic in zip(statistics, expected_tree_statistics):
        value, weight = statistic

        expected_values = expected_statistic["values"]
        expected_weights = expected_statistic["weights"]
        expected_value = numpy.average(expected_values, weights=expected_weights)

        assert math.isclose(value, expected_value)
        assert math.isclose(weight, sum(expected_weights))


@pytest.mark.parametrize("node_index,expected", [(0, [0.5]), (1, [0.25, 0.75]), (2, [0.75, 1.75])])
def test_tree_split_midpoints(
    tree: _FanovaTree, node_index: NodeIndex, expected: List[float]
) -> None:
    numpy.testing.assert_equal(tree._split_midpoints[node_index], expected)


@pytest.mark.parametrize("node_index,expected", [(0, [1.0]), (1, [0.5, 0.5]), (2, [1.5, 0.5])])
def test_tree_split_sizes(tree: _FanovaTree, node_index: NodeIndex, expected: List[float]) -> None:
    numpy.testing.assert_equal(tree._split_sizes[node_index], expected)


@pytest.mark.parametrize(
    "node_index,expected",
    [
        (0, [False, True, True]),
        (1, [False, False, True]),
        (2, [False, False, False]),
        (3, [False, False, False]),
        (4, [False, False, False]),
    ],
)
def test_tree_subtree_active_features(
    tree: _FanovaTree, node_index: NodeIndex, expected: List[bool]
) -> None:
    active_features = tree._subtree_active_features[node_index] == expected  # type: numpy.ndarray
    assert active_features.all()


def test_tree_attrs(tree: _FanovaTree) -> None:
    assert tree._n_features == 3

    assert tree._n_nodes == 5

    assert not tree._is_node_leaf(0)
    assert not tree._is_node_leaf(1)
    assert tree._is_node_leaf(2)
    assert tree._is_node_leaf(3)
    assert tree._is_node_leaf(4)

    assert tree._get_node_left_child(0) == 1
    assert tree._get_node_left_child(1) == 2
    assert tree._get_node_left_child(2) == -1
    assert tree._get_node_left_child(3) == -1
    assert tree._get_node_left_child(4) == -1

    assert tree._get_node_right_child(0) == 4
    assert tree._get_node_right_child(1) == 3
    assert tree._get_node_right_child(2) == -1
    assert tree._get_node_right_child(3) == -1
    assert tree._get_node_right_child(4) == -1

    assert tree._get_node_children(0) == (1, 4)
    assert tree._get_node_children(1) == (2, 3)
    assert tree._get_node_children(2) == (-1, -1)
    assert tree._get_node_children(3) == (-1, -1)
    assert tree._get_node_children(4) == (-1, -1)

    assert tree._get_node_value(0) == -1.0
    assert tree._get_node_value(1) == -1.0
    assert tree._get_node_value(2) == 0.1
    assert tree._get_node_value(3) == 0.2
    assert tree._get_node_value(4) == 0.5

    assert tree._get_node_split_threshold(0) == 0.5
    assert tree._get_node_split_threshold(1) == 1.5
    assert tree._get_node_split_threshold(2) == -1.0
    assert tree._get_node_split_threshold(3) == -1.0
    assert tree._get_node_split_threshold(4) == -1.0

    assert tree._get_node_split_feature(0) == 1
    assert tree._get_node_split_feature(1) == 2


def test_tree_get_node_subspaces(tree: _FanovaTree) -> None:
    search_spaces = numpy.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
    search_spaces_copy = search_spaces.copy()

    # Test splitting on second feature, first node.
    expected_left_child_subspace = numpy.array([[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]])
    expected_right_child_subspace = numpy.array([[0.0, 1.0], [0.5, 1.0], [0.0, 2.0]])

    numpy.testing.assert_array_equal(
        tree._get_node_left_child_subspaces(0, search_spaces), expected_left_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_right_child_subspaces(0, search_spaces), expected_right_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(0, search_spaces)[0], expected_left_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(0, search_spaces)[1], expected_right_child_subspace
    )
    numpy.testing.assert_array_equal(search_spaces, search_spaces_copy)

    # Test splitting on third feature, second node.
    expected_left_child_subspace = numpy.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.5]])
    expected_right_child_subspace = numpy.array([[0.0, 1.0], [0.0, 1.0], [1.5, 2.0]])

    numpy.testing.assert_array_equal(
        tree._get_node_left_child_subspaces(1, search_spaces), expected_left_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_right_child_subspaces(1, search_spaces), expected_right_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(1, search_spaces)[0], expected_left_child_subspace
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(1, search_spaces)[1], expected_right_child_subspace
    )
    numpy.testing.assert_array_equal(search_spaces, search_spaces_copy)
