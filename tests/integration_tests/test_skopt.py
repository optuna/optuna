from typing import Any
from typing import Dict
from typing import List
from unittest.mock import call
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import _pytest.capture
import pytest

import optuna
from optuna import distributions
from optuna._imports import try_import
from optuna.trial import FrozenTrial
from optuna.trial import Trial


with try_import():
    from skopt.space import space

pytestmark = pytest.mark.integration


def test_consider_pruned_trials_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.integration.SkoptSampler(consider_pruned_trials=True)


def test_conversion_from_distribution_to_dimension() -> None:
    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)
    with patch("skopt.Optimizer") as mock_object:
        study.optimize(_objective, n_trials=2, catch=())

        dimensions = [
            # Original: trial.suggest_float('p0', -3.3, 5.2)
            space.Real(-3.3, 5.2),
            # Original: trial.suggest_float('p1', 2.0, 2.0)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.
            # Original: trial.suggest_float('p9', 2.2, 2.2, step=0.5)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.
            # Original: trial.suggest_categorical('p10', ['9', '3', '0', '8'])
            space.Categorical(("9", "3", "0", "8")),
            # Original: trial.suggest_float('p2', 0.0001, 0.3, log=True)
            space.Real(0.0001, 0.3, prior="log-uniform"),
            # Original: trial.suggest_float('p3', 1.1, 1.1, log=True)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.
            # Original: trial.suggest_int('p4', -100, 8)
            space.Integer(0, 108),
            # Original: trial.suggest_int('p5', -20, -20)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.
            # Original: trial.suggest_int('p6', 1, 8, log=True)
            space.Real(0.5, 8.5, prior="log-uniform"),
            # Original: trial.suggest_float('p7', 10, 20, step=2)
            space.Integer(0, 5),
            # Original: trial.suggest_float('p8', 0.1, 1.0, step=0.1)
            space.Integer(0, 8),
        ]

        assert mock_object.mock_calls[0] == call(dimensions)


def test_skopt_kwargs() -> None:
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={"base_estimator": "GBRT"})
    study = optuna.create_study(sampler=sampler)

    with patch("skopt.Optimizer") as mock_object:
        study.optimize(lambda t: t.suggest_int("x", -10, 10), n_trials=2)

        dimensions = [space.Integer(0, 20)]
        assert mock_object.mock_calls[0] == call(dimensions, base_estimator="GBRT")


def test_skopt_kwargs_dimensions() -> None:
    # User specified `dimensions` argument will be ignored in `SkoptSampler`.
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={"dimensions": []})
    study = optuna.create_study(sampler=sampler)

    with patch("skopt.Optimizer") as mock_object:
        study.optimize(lambda t: t.suggest_int("x", -10, 10), n_trials=2)

        expected_dimensions = [space.Integer(0, 20)]
        assert mock_object.mock_calls[0] == call(expected_dimensions)


def test_is_compatible() -> None:
    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)

    study.optimize(lambda t: t.suggest_float("p0", 0, 10), n_trials=1)
    search_space = optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))
    assert search_space == {"p0": distributions.FloatDistribution(low=0, high=10)}

    optimizer = optuna.integration.skopt._Optimizer(search_space, {})

    # Compatible.
    trial = _create_frozen_trial(
        {"p0": 5}, {"p0": distributions.FloatDistribution(low=0, high=10)}
    )
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial(
        {"p0": 5}, {"p0": distributions.FloatDistribution(low=0, high=100)}
    )
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial(
        {"p0": 5, "p1": 7},
        {
            "p0": distributions.FloatDistribution(low=0, high=10),
            "p1": distributions.FloatDistribution(low=0, high=10),
        },
    )
    assert optimizer._is_compatible(trial)

    # Incompatible ('p0' doesn't exist).
    trial = _create_frozen_trial(
        {"p1": 5}, {"p1": distributions.FloatDistribution(low=0, high=10)}
    )
    assert not optimizer._is_compatible(trial)

    # Incompatible (the value of 'p0' is out of range).
    trial = _create_frozen_trial(
        {"p0": 20}, {"p0": distributions.FloatDistribution(low=0, high=100)}
    )
    assert not optimizer._is_compatible(trial)

    # Error (different distribution class).
    trial = _create_frozen_trial({"p0": 5}, {"p0": distributions.IntDistribution(low=0, high=10)})
    with pytest.raises(ValueError):
        optimizer._is_compatible(trial)


def test_warn_independent_sampling(capsys: _pytest.capture.CaptureFixture) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_categorical("x", ["a", "b"])
        if x == "a":
            return trial.suggest_float("y", 0, 1)
        else:
            return trial.suggest_float("z", 0, 1)

    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.enable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.integration.SkoptSampler(warn_independent_sampling=True, n_startup_trials=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    _, err = capsys.readouterr()
    assert err


def _objective(trial: optuna.trial.Trial) -> float:
    p0 = trial.suggest_float("p0", -3.3, 5.2)
    p1 = trial.suggest_float("p1", 2.0, 2.0)
    p2 = trial.suggest_float("p2", 0.0001, 0.3, log=True)
    p3 = trial.suggest_float("p3", 1.1, 1.1, log=True)
    p4 = trial.suggest_int("p4", -100, 8)
    p5 = trial.suggest_int("p5", -20, -20)
    p6 = trial.suggest_int("p6", 1, 8, log=True)
    p7 = trial.suggest_float("p7", 10, 20, step=2)
    p8 = trial.suggest_float("p8", 0.1, 1.0, step=0.1)
    p9 = trial.suggest_float("p9", 2.2, 2.2, step=0.5)
    p10 = trial.suggest_categorical("p10", ["9", "3", "0", "8"])

    return p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + int(p10)


def test_sample_relative_n_startup_trials() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    sampler = optuna.integration.SkoptSampler(
        n_startup_trials=2, independent_sampler=independent_sampler
    )
    study = optuna.create_study(sampler=sampler)

    # The independent sampler is used for Trial#0 and Trial#1.
    # SkoptSampler is used for Trial#2.
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_independent, patch.object(
        sampler, "sample_relative", wraps=sampler.sample_relative
    ) as mock_relative:
        study.optimize(lambda t: t.suggest_int("x", -1, 1) + t.suggest_int("y", -1, 1), n_trials=3)
        assert mock_independent.call_count == 4  # The objective function has two parameters.
        assert mock_relative.call_count == 3


def test_get_trials() -> None:
    with patch(
        "optuna.Study._get_trials",
        new=Mock(side_effect=lambda deepcopy, use_cache: _create_trials()),
    ):
        sampler = optuna.integration.SkoptSampler(consider_pruned_trials=False)
        study = optuna.create_study(sampler=sampler)
        trials = sampler._get_trials(study)
        assert len(trials) == 1

        sampler = optuna.integration.SkoptSampler(consider_pruned_trials=True)
        study = optuna.create_study(sampler=sampler)
        trials = sampler._get_trials(study)
        assert len(trials) == 2
        assert trials[0].value == 1.0
        assert trials[1].value == 2.0


def _create_trials() -> List[FrozenTrial]:
    trials = []
    trials.append(_create_frozen_trial({}, {}))
    trials.append(
        FrozenTrial(
            number=1,
            value=None,
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={},
            system_attrs={},
            params={},
            distributions={},
            intermediate_values={0: 2.0},
            datetime_start=None,
            datetime_complete=None,
            trial_id=0,
        )
    )
    return trials


def _create_frozen_trial(
    params: Dict[str, Any], param_distributions: Dict[str, distributions.BaseDistribution]
) -> FrozenTrial:
    return FrozenTrial(
        number=0,
        value=1.0,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={},
        system_attrs={},
        params=params,
        distributions=param_distributions,
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=0,
    )


def test_call_after_trial_of_independent_sampler() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.integration.SkoptSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        independent_sampler, "after_trial", wraps=independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1
