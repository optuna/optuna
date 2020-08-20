import numpy


from optuna.importance._fanova._fanova import _CategoricalFeaturesOneHotEncoder


def test_categorical_features_one_hot_encoder() -> None:
    # Create test data with 5 columns with the following types of features.
    # 0: Numerical
    # 1: Categorical (3 categories)
    # 2: Numerical
    # 3: Caterogical (4 categories)
    # 4: Numerical
    X = numpy.array(
        [[1.0, 0.0, 1.0, 3.0, 2.0], [2.0, 1.0, 2.0, 2.0, 3.0], [3.0, 2.0, 1.0, 2.0, 0.0]],
        dtype=numpy.float64,
    )
    search_spaces = numpy.array(
        [[1.0, 4.0], [0.0, 3.0], [1.0, 3], [0.0, 4.0], [0.0, 4.0]], dtype=numpy.float64
    )
    search_spaces_is_categorical = [False, True, False, True, False]

    encoder = _CategoricalFeaturesOneHotEncoder()
    X, search_spaces = encoder.fit_transform(X, search_spaces, search_spaces_is_categorical)

    assert X.shape == (3, 10)
    assert X[:, :7].min() == 0.0  # First 3 + 4 columns are one-hot encoded categorical.
    assert X[:, :7].max() == 1.0

    numpy.testing.assert_array_equal(
        search_spaces,
        numpy.array(
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 4], [1, 3], [0, 4]],
            dtype=numpy.float64,
        ),
    )

    features_to_raw_features = encoder.features_to_raw_features
    assert features_to_raw_features is not None  # Encoder is fitted.
    expected_features_to_raw_features = [
        [7],
        [0, 1, 2],
        [8],
        [3, 4, 5, 6],
        [9],
    ]
    for actual, expected in zip(features_to_raw_features, expected_features_to_raw_features):
        numpy.testing.assert_array_equal(actual, expected)
