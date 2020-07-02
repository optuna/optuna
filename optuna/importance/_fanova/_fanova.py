"""An implementation of `An Efficient Approach for Assessing Hyperparameter Importance`.

See http://proceedings.mlr.press/v32/hutter14.pdf and https://automl.github.io/fanova/cite.html
for how to cite the original work.

This implementation is inspired by the efficient algorithm in
`fanova` (https://github.com/automl/fanova) and
`pyrfr` (https://github.com/automl/random_forest_run) by the original authors.

Differences include relying on scikit-learn to fit random forests
(`sklearn.ensemble.RandomForestRegressor`) and that it is otherwise written entirely in Python.
This stands in contrast to the original implementation which is partially written in C++.
Since Python runtime overhead may become noticeable, included are instead several
optimizations, e.g. vectorized NumPy functions to compute the marginals, instead of keeping all
running statistics. Known cases include assessing higher order importances, e.g. pairwise
importances, this is due to the fact that the number of partitions to visit grows exponentially,
or when assessing categorical features with a larger number of choices since each choice is
given a unique one-hot encoded raw feature.

"""

import itertools
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy

from optuna._imports import try_import
from optuna.importance._fanova._tree import _FanovaTree

with try_import() as _imports:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder


class _Fanova(object):
    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        min_samples_split: Union[int, float],
        min_samples_leaf: Union[int, float],
        seed: Optional[int],
    ) -> None:
        _imports.check()

        self._forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        self._trees = None  # type: Optional[List[_FanovaTree]]
        self._variances = None  # type: Optional[Dict[Tuple[int, ...], numpy.ndarray]]
        self._features_to_raw_features = None  # type: Optional[List[numpy.ndarray]]

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
    ) -> None:
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == search_spaces.shape[0]
        assert X.shape[1] == len(search_spaces_is_categorical)
        assert search_spaces.shape[1] == 2

        encoder = _CategoricalFeaturesOneHotEncoder()
        X, search_spaces = encoder.fit_transform(X, search_spaces, search_spaces_is_categorical)

        self._forest.fit(X, y)

        self._trees = [_FanovaTree(e.tree_, search_spaces) for e in self._forest.estimators_]
        self._features_to_raw_features = encoder.features_to_raw_features
        self._variances = {}

        if all(tree.variance == 0 for tree in self._trees):
            # If all trees have 0 variance, we cannot assess any importances.
            # This could occur if for instance `X.shape[0] == 1`.
            raise RuntimeError("Encountered zero total variance in all trees.")

    def get_importance(self, features: Tuple[int, ...]) -> Tuple[float, float]:
        # Assert that `fit` has been called.
        assert self._trees is not None
        assert self._variances is not None

        self._compute_variances(features)

        fractions = []  # type: Union[List[float], numpy.ndarray]

        for tree_index, tree in enumerate(self._trees):
            tree_variance = tree.variance
            if tree_variance > 0.0:
                fraction = self._variances[features][tree_index] / tree_variance
                fractions.append(fraction)

        fractions = numpy.array(fractions)

        return fractions.mean(), fractions.std()

    def _compute_variances(self, features: Tuple[int, ...]) -> None:
        assert self._trees is not None
        assert self._variances is not None
        assert self._features_to_raw_features is not None

        if features in self._variances:
            return

        for k in range(1, len(features)):
            for sub_features in itertools.combinations(features, k):
                if sub_features not in self._variances:
                    self._compute_variances(sub_features)

        raw_features = numpy.concatenate([self._features_to_raw_features[f] for f in features])

        variances = numpy.empty(len(self._trees), dtype=numpy.float64)

        for tree_index, tree in enumerate(self._trees):
            marginal_variance = tree.get_marginal_variance(raw_features)

            # See `fANOVA.__compute_marginals` in
            # https://github.com/automl/fanova/blob/master/fanova/fanova.py.
            for k in range(1, len(features)):
                for sub_features in itertools.combinations(features, k):
                    marginal_variance -= self._variances[sub_features][tree_index]

            variances[tree_index] = numpy.clip(marginal_variance, 0.0, numpy.inf)

        self._variances[features] = variances


class _CategoricalFeaturesOneHotEncoder(object):
    def __init__(self) -> None:
        # `features_to_raw_features["column index in original matrix"]
        #     == "numpy.ndarray with corresponding columns in the transformed matrix"`
        self.features_to_raw_features = None  # type: Optional[List[numpy.ndarray]]

    def fit_transform(
        self,
        X: numpy.ndarray,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Transform the `X` matrix by expanding categorical integer-valued columns to one-hot
        # encoding matrices and search spaces `search_spaces` similarly.
        # Note that the resulting matrices are sparse and potentially very big.

        n_features = X.shape[1]
        assert n_features == len(search_spaces)
        assert n_features == len(search_spaces_is_categorical)

        categories = []
        categorical_features = []
        categorical_features_n_uniques = {}
        numerical_features = []

        for feature, is_categorical in enumerate(search_spaces_is_categorical):
            if is_categorical:
                n_unique = search_spaces[feature][1].astype(numpy.int32)
                categories.append(numpy.arange(n_unique))
                categorical_features.append(feature)
                categorical_features_n_uniques[feature] = n_unique
            else:
                numerical_features.append(feature)

        transformer = ColumnTransformer(
            [
                (
                    "_categorical",
                    OneHotEncoder(categories=categories, sparse=False),
                    categorical_features,
                )
            ],
            remainder="passthrough",
        )

        # All categorical one-hot features will be placed before the numerical features in
        # `ColumnTransformer.fit_transform`.
        X = transformer.fit_transform(X)

        features_to_raw_features = [None for _ in range(n_features)]  # type: List[numpy.ndarray]
        i = 0
        if len(categorical_features) > 0:
            categories = transformer.transformers_[0][1].categories_
            assert len(categories) == len(categorical_features)

            for j, (feature, category) in enumerate(zip(categorical_features, categories)):
                categorical_raw_features = category.astype(numpy.int32)
                if i > 0:
                    # Adjust offset.
                    previous_categorical_feature = categorical_features[j - 1]
                    previous_categorical_raw_features = features_to_raw_features[
                        previous_categorical_feature
                    ]
                    categorical_raw_features += previous_categorical_raw_features[-1] + 1
                assert features_to_raw_features[feature] is None
                features_to_raw_features[feature] = categorical_raw_features
                i = categorical_raw_features[-1] + 1
        for feature in numerical_features:
            features_to_raw_features[feature] = numpy.atleast_1d(i)
            i += 1
        assert i == X.shape[1]

        # `raw_features_to_features["column index in transformed matrix"]
        #     == "column in the original matrix"`
        raw_features_to_features = numpy.empty((X.shape[1],), dtype=numpy.int32)
        i = 0
        for categorical_feature in categorical_features:
            for _ in range(categorical_features_n_uniques[categorical_feature]):
                raw_features_to_features[i] = categorical_feature
                i += 1
        for numerical_col in numerical_features:
            raw_features_to_features[i] = numerical_col
            i += 1
        assert i == X.shape[1]

        # Transform search spaces.
        n_raw_features = raw_features_to_features.size
        raw_search_spaces = numpy.empty((n_raw_features, 2), dtype=numpy.float64)

        for raw_feature, feature in enumerate(raw_features_to_features):
            if search_spaces_is_categorical[feature]:
                raw_search_spaces[raw_feature] = [0.0, 1.0]
            else:
                raw_search_spaces[raw_feature] = search_spaces[feature]

        self.features_to_raw_features = features_to_raw_features

        return X, raw_search_spaces
