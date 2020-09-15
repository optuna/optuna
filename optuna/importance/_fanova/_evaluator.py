from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_study_data
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.study import Study


class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA importance evaluator.

    Implements the fANOVA hyperparameter importance evaluation algorithm in
    `An Efficient Approach for Assessing Hyperparameter Importance
    <http://proceedings.mlr.press/v32/hutter14.html>`_.

    Given a study, fANOVA fits a random forest regression model that predicts the objective value
    given a parameter configuration. The more accurate this model is, the more reliable the
    importances assessed by this class are.

    .. note::

        Requires the `sklearn <https://github.com/scikit-learn/scikit-learn>`_ Python package.

    .. note::

        Pairwise and higher order importances are not supported through this class. They can be
        computed using :class:`~optuna.importance._fanova._fanova._Fanova` directly but is not
        recommended as interfaces may change without prior notice.

    .. note::

        The performance of fANOVA depends on the prediction performance of the underlying
        random forest model. In order to obtain high prediction performance, it is necessary to
        cover a wide range of the hyperparameter search space. It is recommended to use an
        exploration-oriented sampler such as :class:`~optuna.samplers.RandomSampler`.

    .. note::

        For how to cite the original work, please refer to
        https://automl.github.io/fanova/cite.html.

    Args:
        n_trees:
            The number of trees in the forest.
        max_depth:
            The maximum depth of the trees in the forest.
        seed:
            Controls the randomness of the forest. For deterministic behavior, specify a value
            other than :obj:`None`.

    """

    def __init__(
        self, *, n_trees: int = 64, max_depth: int = 64, seed: Optional[int] = None
    ) -> None:
        self._evaluator = _Fanova(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=seed,
        )

    def evaluate(self, study: Study, params: Optional[List[str]] = None) -> Dict[str, float]:
        distributions = _get_distributions(study, params)
        params_data, values_data = _get_study_data(study, distributions)

        if params_data.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        # Many (deep) copies of the search spaces are required during the tree traversal and using
        # Optuna distributions will create a bottleneck.
        # Therefore, search spaces (parameter distributions) are represented by a single
        # `numpy.ndarray`, coupled with a list of flags that indicate whether they are categorical
        # or not.
        search_spaces = numpy.empty((len(distributions), 2), dtype=numpy.float64)
        search_spaces_is_categorical = []

        for i, distribution in enumerate(distributions.values()):
            if isinstance(distribution, CategoricalDistribution):
                search_spaces[i, 0] = 0
                search_spaces[i, 1] = len(distribution.choices)
                search_spaces_is_categorical.append(True)
            elif isinstance(
                distribution,
                (
                    DiscreteUniformDistribution,
                    IntLogUniformDistribution,
                    IntUniformDistribution,
                    LogUniformDistribution,
                    UniformDistribution,
                ),
            ):
                search_spaces[i, 0] = distribution.low
                search_spaces[i, 1] = distribution.high
                search_spaces_is_categorical.append(False)
            else:
                assert False

        evaluator = self._evaluator
        evaluator.fit(
            X=params_data,
            y=values_data,
            search_spaces=search_spaces,
            search_spaces_is_categorical=search_spaces_is_categorical,
        )

        importances = {}
        for i, name in enumerate(distributions.keys()):
            importance, _ = evaluator.get_importance((i,))
            importances[name] = importance

        total_importance = sum(importances.values())
        for name in importances.keys():
            importances[name] /= total_importance

        sorted_importances = OrderedDict(
            reversed(
                sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
            )
        )
        return sorted_importances
