import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy

from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.trial import FrozenTrial


with try_import() as _imports:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder


class _Transform:
    def __init__(
        self,
        trials: List[FrozenTrial],
        search_space: Dict[str, BaseDistribution],
        transform_log: bool = True,
    ) -> None:
        _imports.check()

        self._search_space = search_space
        self._transform_log = transform_log
        self._encoder = _CategoricalOneHotEncoder()

        n_trials = len(trials)
        n_params = len(search_space)

        assert n_trials > 0, "Cannot transform if no trials are given."
        assert n_params > 0, "Cannot transform if no distributions are given."

        # TODO(hvy): Allocate transformed (one-hot encoded) memory at once, instead of first
        # creating a naive array and then constructing a new one using the encoder.
        params = numpy.empty((n_trials, n_params), dtype=numpy.float64)
        values = numpy.empty((n_trials,), dtype=numpy.float64)
        bounds = numpy.empty((n_params, 2), dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            for distribution_idx, (name, distribution) in enumerate(search_space.items()):
                params[trial_idx, distribution_idx] = self._transform_param(
                    trial.params[name], distribution
                )
            values[trial_idx] = trial.value

        for distribution_idx, distribution in enumerate(search_space.values()):
            bounds[distribution_idx] = self._transform_distribution_bounds(distribution)

        bounds_is_categorical = []
        for distribution in search_space.values():
            if isinstance(distribution, CategoricalDistribution):
                bounds_is_categorical.append(True)
            else:
                bounds_is_categorical.append(False)

        params, bounds = self._encoder.fit_transform(params, bounds, bounds_is_categorical)

        self._params = params
        self._bounds = bounds
        self._values = values

    @property
    def params(self) -> numpy.ndarray:
        return self._params

    @property
    def bounds(self) -> numpy.ndarray:
        return self._bounds

    @property
    def values(self) -> numpy.ndarray:
        return self._values

    def untransform(self, transformed_params: numpy.ndarray) -> Dict[str, Any]:
        assert self._encoder.cols_to_encoded_cols is not None
        assert transformed_params.shape == (self.params.shape[1],)

        params = {}

        for i, (name, distribution) in enumerate(self._search_space.items()):
            transformed_param = transformed_params[self._encoder.cols_to_encoded_cols[i]]
            params[name] = self._untransform_param(transformed_param, distribution)

        return params

    def _transform_param(self, param: Any, distribution: BaseDistribution) -> float:
        d = distribution
        if isinstance(d, CategoricalDistribution):
            trans_param = float(d.to_internal_repr(param))
        elif isinstance(d, UniformDistribution):
            trans_param = float(param)
        elif isinstance(d, LogUniformDistribution):
            trans_param = math.log(param) if self._transform_log else float(param)
        elif isinstance(d, DiscreteUniformDistribution):
            trans_param = float(param)
        elif isinstance(d, IntUniformDistribution):
            trans_param = float(param)
        elif isinstance(d, IntLogUniformDistribution):
            trans_param = math.log(param) if self._transform_log else float(param)
        else:
            assert False
        return trans_param

    def _transform_distribution_bounds(
        self, distribution: BaseDistribution
    ) -> Tuple[float, float]:
        if isinstance(distribution, CategoricalDistribution):
            bounds = (0.0, float(len(distribution.choices)))
        elif isinstance(
            distribution,
            (
                UniformDistribution,
                LogUniformDistribution,
                DiscreteUniformDistribution,
                IntUniformDistribution,
                IntLogUniformDistribution,
            ),
        ):
            # TODO(hvy): Allow subtracting eps from high to use this class from `CmaEsSampler`.
            bounds = (
                self._transform_param(distribution.low, distribution),
                self._transform_param(distribution.high, distribution),
            )
        else:
            assert False
        return bounds

    def _untransform_param(
        self, transformed_param: numpy.ndarray, distribution: BaseDistribution
    ) -> Any:
        d = distribution
        if isinstance(d, CategoricalDistribution):
            # Select the highest rated one-hot enconding.
            param = d.to_external_repr(transformed_param.argmax())
        elif isinstance(d, UniformDistribution):
            param = float(transformed_param.item())
        elif isinstance(d, LogUniformDistribution):
            param = (
                math.exp(transformed_param) if self._transform_log else float(transformed_param)
            )
        elif isinstance(d, DiscreteUniformDistribution):
            v = numpy.round(transformed_param / d.q) * d.q + d.low
            # v may slightly exceed range due to round-off errors.
            param = float(min(max(v, d.low), d.high))
        elif isinstance(d, IntUniformDistribution):
            r = numpy.round((transformed_param - d.low) / d.step)
            v = r * d.step + d.low
            param = int(v)
        elif isinstance(d, IntLogUniformDistribution):
            if self._transform_log:
                param = math.exp(transformed_param)
                v = numpy.round(param)
                param = int(min(max(v, d.low), d.high))
            else:
                param = int(transformed_param)
        else:
            assert False
        return param


class _CategoricalOneHotEncoder:
    # `cols_to_encoded_cols["column index in original matrix"]
    #     == "numpy.ndarray with corresponding columns in the encoded matrix"`
    cols_to_encoded_cols: Optional[List[numpy.ndarray]]

    # `encoded_cols_to_cols["column index in encoded matrix"]
    #     == "column index in the original matrix"`
    encoded_cols_to_cols: Optional[numpy.ndarray]

    def fit_transform(
        self,
        params: numpy.ndarray,
        bounds: numpy.ndarray,
        bounds_is_categorical: List[bool],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Transform the `params` matrix by expanding categorical integer-valued columns to one-hot
        # encoding matrices and search spaces `bounds` similarly.
        # Note that the resulting matrices are sparse and potentially very big.

        n_cols = params.shape[1]
        assert n_cols == len(bounds)
        assert n_cols == len(bounds_is_categorical)

        categories = []
        categorical_cols = []
        categorical_cols_n_unique = {}
        numerical_cols = []

        for col, is_categorical in enumerate(bounds_is_categorical):
            if is_categorical:
                n_unique = bounds[col][1].astype(numpy.int32)
                categories.append(numpy.arange(n_unique))
                categorical_cols.append(col)
                categorical_cols_n_unique[col] = n_unique
            else:
                numerical_cols.append(col)

        transformer = ColumnTransformer(
            [
                (
                    "_categorical",
                    OneHotEncoder(categories=categories, sparse=False),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )

        # All categorical one-hot columns will be placed before the numerical columns in
        # `ColumnTransformer.fit_transform`.
        params = transformer.fit_transform(params)

        cols_to_encoded_cols: List[numpy.ndarray] = [None for _ in range(n_cols)]
        i = 0
        if len(categorical_cols) > 0:
            categories = transformer.transformers_[0][1].categories_
            assert len(categories) == len(categorical_cols)

            for j, (col, category) in enumerate(zip(categorical_cols, categories)):
                categorical_encoded_cols = category.astype(numpy.int32)
                if i > 0:
                    # Adjust offset.
                    previous_categorical_col = categorical_cols[j - 1]
                    previous_categorical_encoded_cols = cols_to_encoded_cols[
                        previous_categorical_col
                    ]
                    categorical_encoded_cols += previous_categorical_encoded_cols[-1] + 1
                assert cols_to_encoded_cols[col] is None
                cols_to_encoded_cols[col] = categorical_encoded_cols
                i = categorical_encoded_cols[-1] + 1
        for col in numerical_cols:
            cols_to_encoded_cols[col] = numpy.atleast_1d(i)
            i += 1
        assert i == params.shape[1]

        encoded_cols_to_cols = numpy.empty((params.shape[1],), dtype=numpy.int32)
        i = 0
        for categorical_feature in categorical_cols:
            for _ in range(categorical_cols_n_unique[categorical_feature]):
                encoded_cols_to_cols[i] = categorical_feature
                i += 1
        for numerical_col in numerical_cols:
            encoded_cols_to_cols[i] = numerical_col
            i += 1
        assert i == params.shape[1]

        # Transform search spaces.
        n_raw_features = encoded_cols_to_cols.size
        encoded_bounds = numpy.empty((n_raw_features, 2), dtype=numpy.float64)

        for raw_feature, col in enumerate(encoded_cols_to_cols):
            if bounds_is_categorical[col]:
                encoded_bounds[raw_feature] = [0.0, 1.0]
            else:
                encoded_bounds[raw_feature] = bounds[col]

        self.cols_to_encoded_cols = cols_to_encoded_cols
        self.encoded_cols_to_cols = encoded_cols_to_cols

        return params, encoded_bounds
