from __future__ import annotations

import numpy as np
import pytest

import optuna
from optuna._gp.search_space import get_search_space_and_normalized_params
from optuna._gp.search_space import get_unnormalized_param
from optuna._gp.search_space import normalize_one_param
from optuna._gp.search_space import round_one_normalized_param
from optuna._gp.search_space import sample_normalized_params
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace
from optuna._gp.search_space import unnormalize_one_param
from optuna._transform import _SearchSpaceTransform


@pytest.mark.parametrize(
    "scale_type,bounds,step,unnormalized,normalized",
    [
        (ScaleType.LINEAR, (0.0, 10.0), 0.0, 2.0, 0.2),
        (ScaleType.LINEAR, (0.0, 9.0), 1.0, 2.0, 0.25),
        (ScaleType.LINEAR, (0.5, 8.5), 2.0, 2.5, 0.3),
        (ScaleType.LINEAR, (0.0, 0.0), 0.0, 0.0, 0.5),
        (ScaleType.LOG, (10**0.0, 10**10.0), 0.0, 10**2.0, 0.2),
        (
            ScaleType.LOG,
            (1.0, 10.0),
            1.0,
            2.0,
            (np.log(2.0) - np.log(0.5)) / (np.log(10.5) - np.log(0.5)),
        ),
        (ScaleType.CATEGORICAL, (0.0, 10.0), 0.0, 3.0, 3.0),
    ],
)
def test_normalize_unnormalize_one_param(
    scale_type: ScaleType,
    bounds: tuple[float, float],
    step: float,
    unnormalized: float,
    normalized: float,
) -> None:
    assert np.isclose(
        normalize_one_param(
            np.array(unnormalized),
            scale_type,
            bounds,
            step,
        ),
        normalized,
    )
    assert np.isclose(
        unnormalize_one_param(
            np.array(normalized),
            scale_type,
            bounds,
            step,
        ),
        unnormalized,
    )


@pytest.mark.parametrize(
    "scale_type,bounds,step,value,expected",
    [
        (ScaleType.LINEAR, (0.0, 9.0), 1.0, 0.21, 0.25),
        (
            ScaleType.LOG,
            (1.0, 10.0),
            1.0,
            (np.log(1.8) - np.log(0.5)) / (np.log(10.5) - np.log(0.5)),
            (np.log(2.0) - np.log(0.5)) / (np.log(10.5) - np.log(0.5)),
        ),
        (ScaleType.LINEAR, (-1, 1), 0.5, 0.0, 0.1),
        (ScaleType.LINEAR, (-1, 1), 0.5, 1.0, 0.9),
        (ScaleType.LINEAR, (-0.1, 0.7), 0.4, -0.1, 1 / 6),
        (ScaleType.LINEAR, (-0.1, 0.7), 0.4, 0.7, 5 / 6),
    ],
)
def test_round_one_normalized_param(
    scale_type: ScaleType, bounds: tuple[float, float], step: float, value: float, expected: float
) -> None:
    res = round_one_normalized_param(
        np.array(value),
        scale_type,
        bounds,
        step,
    )
    assert np.isclose(res, expected)
    assert 0.0 <= res <= 1.0


def test_sample_normalized_params() -> None:
    search_space = SearchSpace(
        scale_types=np.array(
            [
                ScaleType.LINEAR,
                ScaleType.LINEAR,
                ScaleType.LOG,
                ScaleType.LOG,
                ScaleType.CATEGORICAL,
            ]
        ),
        bounds=np.array([(0.0, 10.0), (1.0, 10.0), (10.0, 100.0), (10.0, 100.0), (0.0, 5.0)]),
        steps=np.array([0.0, 1.0, 0.0, 1.0, 1.0]),
    )
    samples = sample_normalized_params(
        n=128, search_space=search_space, rng=np.random.RandomState(0)
    )
    assert samples.shape == (128, 5)
    assert np.all((samples[:, :4] >= 0.0) & (samples[:, :4] <= 1.0))

    integer_params = [1, 3, 4]
    for i in integer_params:
        params = unnormalize_one_param(
            samples[:, i],
            search_space.scale_types[i],
            search_space.bounds[i],
            search_space.steps[i],
        )
        # assert params are close to integers
        assert np.allclose((params + 0.5) % 1.0, 0.5)


def test_get_search_space_and_normalized_params_no_categorical() -> None:
    optuna_search_space = {
        "a": optuna.distributions.FloatDistribution(0.0, 10.0),
        "b": optuna.distributions.IntDistribution(0, 10),
        "c": optuna.distributions.FloatDistribution(1.0, 10.0, log=True),
        "d": optuna.distributions.IntDistribution(1, 10, log=True),
        "e": optuna.distributions.CategoricalDistribution(["x", "y", "z"]),
    }
    trials = [
        optuna.create_trial(
            params={"a": 2.0, "b": 2, "c": 2.0, "d": 2, "e": "x"},
            distributions=optuna_search_space,
            value=0.0,
        )
    ]

    search_space, normalized_params = get_search_space_and_normalized_params(
        trials, optuna_search_space
    )
    assert np.all(
        search_space.scale_types
        == np.array(
            [
                ScaleType.LINEAR,
                ScaleType.LINEAR,
                ScaleType.LOG,
                ScaleType.LOG,
                ScaleType.CATEGORICAL,
            ]
        )
    )
    assert np.all(
        search_space.bounds
        == np.array([(0.0, 10.0), (0.0, 10.0), (1.0, 10.0), (1.0, 10.0), (0.0, 3.0)])
    )
    assert np.all(search_space.steps == np.array([0.0, 1.0, 0.0, 1.0, 1.0]))

    non_categorical_search_space = {
        param: dist
        for param, dist in optuna_search_space.items()
        if not isinstance(dist, optuna.distributions.CategoricalDistribution)
    }
    search_space_transform = _SearchSpaceTransform(
        search_space=non_categorical_search_space,
        transform_log=True,
        transform_step=True,
        transform_0_1=True,
    )
    expected = search_space_transform.transform(trials[0].params)
    assert np.allclose(normalized_params[:, :4], expected)
    assert normalized_params[0, 4] == 0.0


def test_get_untransform_search_space() -> None:
    optuna_search_space = {
        "a": optuna.distributions.FloatDistribution(0.0, 10.0),
        "b": optuna.distributions.IntDistribution(0, 9),
        "c": optuna.distributions.FloatDistribution(2.0**0, 2.0**10, log=True),
        "d": optuna.distributions.IntDistribution(1, 10, log=True),
        "e": optuna.distributions.CategoricalDistribution(["x", "y", "z"]),
    }

    normalized_values = np.array(
        [
            0.25,
            0.25,
            0.5,
            (np.log(2.0) - np.log(0.5)) / (np.log(10.5) - np.log(0.5)),
            0.0,
        ]
    )
    params = get_unnormalized_param(optuna_search_space, normalized_values)

    expected = {
        "a": 2.5,
        "b": 2,
        "c": 2.0**5,
        "d": 2,
        "e": "x",
    }
    assert params == expected
