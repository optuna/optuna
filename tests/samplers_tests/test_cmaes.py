import pickle
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
import warnings

from cmaes import CMA
import numpy as np
import pytest

import optuna
from optuna import create_trial
from optuna._transform import _SearchSpaceTransform
from optuna.samplers._cmaes import _concat_optimizer_attrs
from optuna.samplers._cmaes import _split_optimizer_str
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_consider_pruned_trials_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.CmaEsSampler(consider_pruned_trials=True)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize(
    "use_separable_cma, cma_class_str",
    [(False, "optuna.samplers._cmaes.CMA"), (True, "optuna.samplers._cmaes.SepCMA")],
)
def test_init_cmaes_opts(use_separable_cma: bool, cma_class_str: str) -> None:
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        independent_sampler=DeterministicRelativeSampler({}, {}),
        use_separable_cma=use_separable_cma,
    )
    study = optuna.create_study(sampler=sampler)

    with patch(cma_class_str) as cma_class:
        cma_obj = MagicMock()
        cma_obj.ask.return_value = np.array((-1, -1))
        cma_obj.generation = 0
        cma_obj.population_size = 5
        cma_class.return_value = cma_obj
        study.optimize(
            lambda t: t.suggest_float("x", -1, 1) + t.suggest_float("y", -1, 1), n_trials=2
        )

        assert cma_class.call_count == 1

        _, actual_kwargs = cma_class.call_args
        assert np.array_equal(actual_kwargs["mean"], np.array([0, 0]))
        assert actual_kwargs["sigma"] == 0.1
        assert np.allclose(actual_kwargs["bounds"], np.array([(-1, 1), (-1, 1)]))
        assert actual_kwargs["seed"] == np.random.RandomState(1).randint(1, 2 ** 32)
        assert actual_kwargs["n_max_resampling"] == 10 * 2
        assert actual_kwargs["population_size"] is None


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@patch("optuna.samplers._cmaes.get_warm_start_mgd")
def test_warm_starting_cmaes(mock_func_ws: MagicMock) -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x ** 2 + y

    source_study = optuna.create_study()
    source_study.optimize(objective, 20)
    source_trials = source_study.get_trials(deepcopy=False)

    mock_func_ws.return_value = (np.zeros(2), 0.0, np.zeros((2, 2)))
    sampler = optuna.samplers.CmaEsSampler(seed=1, n_startup_trials=1, source_trials=source_trials)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, 2)
    assert mock_func_ws.call_count == 1


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_should_raise_exception() -> None:
    dummy_source_trials = [create_trial(value=i, state=TrialState.COMPLETE) for i in range(10)]

    with pytest.raises(ValueError):
        optuna.samplers.CmaEsSampler(
            x0={"x": 0.1, "y": 0.1},
            source_trials=dummy_source_trials,
        )

    with pytest.raises(ValueError):
        optuna.samplers.CmaEsSampler(
            sigma0=0.1,
            source_trials=dummy_source_trials,
        )

    with pytest.raises(ValueError):
        optuna.samplers.CmaEsSampler(
            use_separable_cma=True,
            source_trials=dummy_source_trials,
        )

    with pytest.raises(ValueError):
        optuna.samplers.CmaEsSampler(
            restart_strategy="invalid-restart-strategy",
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_incompatible_search_space() -> None:
    def objective1(trial: optuna.Trial) -> float:
        x0 = trial.suggest_float("x0", 2, 3)
        x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
        return x0 + x1

    source_study = optuna.create_study()
    source_study.optimize(objective1, 20)

    # Should not raise an exception.
    sampler = optuna.samplers.CmaEsSampler(source_trials=source_study.trials)
    target_study1 = optuna.create_study(sampler=sampler)
    target_study1.optimize(objective1, 20)

    def objective2(trial: optuna.Trial) -> float:
        x0 = trial.suggest_float("x0", 2, 3)
        x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
        x2 = trial.suggest_float("x2", 1e-2, 1e2, log=True)
        return x0 + x1 + x2

    # Should raise an exception.
    sampler = optuna.samplers.CmaEsSampler(source_trials=source_study.trials)
    target_study2 = optuna.create_study(sampler=sampler)
    with pytest.raises(ValueError):
        target_study2.optimize(objective2, 20)


def test_infer_relative_search_space_1d() -> None:
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler)

    # The distribution has only one candidate.
    study.optimize(lambda t: t.suggest_int("x", 1, 1), n_trials=1)
    assert sampler.infer_relative_search_space(study, study.best_trial) == {}


def test_sample_relative_1d() -> None:
    independent_sampler = DeterministicRelativeSampler({}, {})
    sampler = optuna.samplers.CmaEsSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)

    # If search space is one dimensional, the independent sampler is always used.
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_object:
        study.optimize(lambda t: t.suggest_int("x", -1, 1), n_trials=2)
        assert mock_object.call_count == 2


def test_sample_relative_n_startup_trials() -> None:
    independent_sampler = DeterministicRelativeSampler({}, {})
    sampler = optuna.samplers.CmaEsSampler(
        n_startup_trials=2, independent_sampler=independent_sampler
    )
    study = optuna.create_study(sampler=sampler)

    def objective(t: optuna.Trial) -> float:

        value = t.suggest_int("x", -1, 1) + t.suggest_int("y", -1, 1)
        if t.number == 0:
            raise Exception("first trial is failed")
        return float(value)

    # The independent sampler is used for Trial#0 (FAILED), Trial#1 (COMPLETE)
    # and Trial#2 (COMPLETE). The CMA-ES is used for Trial#3 (COMPLETE).
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_independent, patch.object(
        sampler, "sample_relative", wraps=sampler.sample_relative
    ) as mock_relative:
        study.optimize(objective, n_trials=4, catch=(Exception,))
        assert mock_independent.call_count == 6  # The objective function has two parameters.
        assert mock_relative.call_count == 4


def test_reseed_rng() -> None:
    sampler = optuna.samplers.CmaEsSampler()

    with patch.object(
        sampler._independent_sampler, "reseed_rng", wraps=sampler._independent_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1


def test_get_trials() -> None:
    with patch("optuna.Study.get_trials", new=Mock(side_effect=lambda deepcopy: _create_trials())):
        sampler = optuna.samplers.CmaEsSampler(consider_pruned_trials=False)
        study = optuna.create_study(sampler=sampler)
        trials = sampler._get_trials(study)
        assert len(trials) == 1

        sampler = optuna.samplers.CmaEsSampler(consider_pruned_trials=True)
        study = optuna.create_study(sampler=sampler)
        trials = sampler._get_trials(study)
        assert len(trials) == 2
        assert trials[0].value == 1.0
        assert trials[1].value == 2.0


def _create_trials() -> List[FrozenTrial]:
    trials = []
    trials.append(
        FrozenTrial(
            number=0,
            value=1.0,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={},
            system_attrs={},
            params={},
            distributions={},
            intermediate_values={},
            datetime_start=None,
            datetime_complete=None,
            trial_id=0,
        )
    )
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


def test_population_size_is_multiplied_when_enable_ipop() -> None:
    inc_popsize = 2
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        independent_sampler=DeterministicRelativeSampler({}, {}),
        restart_strategy="ipop",
        inc_popsize=inc_popsize,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        _ = trial.suggest_float("x", -1, 1)
        _ = trial.suggest_float("y", -1, 1)
        return 1.0

    with patch("optuna.samplers._cmaes.CMA") as cma_class_mock, patch(
        "optuna.samplers._cmaes.pickle"
    ) as pickle_mock:
        pickle_mock.dump.return_value = b"serialized object"

        should_stop_mock = MagicMock()
        should_stop_mock.return_value = True

        cma_obj = CMA(
            mean=np.array([-1, -1], dtype=float),
            sigma=1.3,
            bounds=np.array([[-1, 1], [-1, 1]], dtype=float),
        )
        cma_obj.should_stop = should_stop_mock
        cma_class_mock.return_value = cma_obj

        popsize = cma_obj.population_size
        study.optimize(objective, n_trials=2 + popsize)
        assert cma_obj.should_stop.call_count == 1

        _, actual_kwargs = cma_class_mock.call_args
        assert actual_kwargs["population_size"] == inc_popsize * popsize


def test_restore_optimizer_keeps_backward_compatibility() -> None:
    sampler = optuna.samplers.CmaEsSampler()
    optimizer = CMA(np.zeros(2), sigma=1.3)
    optimizer_str = pickle.dumps(optimizer).hex()

    completed_trials = [
        create_trial(state=TrialState.COMPLETE, value=0.1),
        create_trial(
            state=TrialState.COMPLETE,
            value=0.1,
            system_attrs={"cma:optimizer": optimizer_str, "cma:n_restarts": 1},
        ),
        create_trial(state=TrialState.COMPLETE, value=0.1),
    ]
    optimizer, n_restarts = sampler._restore_optimizer(completed_trials)
    assert isinstance(optimizer, CMA)
    assert n_restarts == 1


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_restore_optimizer_from_substrings() -> None:
    sampler = optuna.samplers.CmaEsSampler()
    optimizer = CMA(np.zeros(10), sigma=1.3)
    optimizer_str = pickle.dumps(optimizer).hex()

    system_attrs: Dict[str, Any] = _split_optimizer_str(optimizer_str)
    assert len(system_attrs) > 1
    system_attrs["cma:n_restarts"] = 1

    completed_trials = [
        create_trial(state=TrialState.COMPLETE, value=0.1),
        create_trial(
            state=TrialState.COMPLETE,
            value=0.1,
            system_attrs=system_attrs,
        ),
        create_trial(state=TrialState.COMPLETE, value=0.1),
    ]
    optimizer, n_restarts = sampler._restore_optimizer(completed_trials)
    assert isinstance(optimizer, CMA)
    assert n_restarts == 1


@pytest.mark.parametrize(
    "dummy_optimizer_str,attr_len",
    [
        ("012", 1),
        ("01234", 1),
        ("012345", 2),
    ],
)
def test_split_and_concat_optimizer_string(dummy_optimizer_str: str, attr_len: int) -> None:
    with patch("optuna.samplers._cmaes._SYSTEM_ATTR_MAX_LENGTH", 5):
        attrs = _split_optimizer_str(dummy_optimizer_str)
        assert len(attrs) == attr_len
        actual = _concat_optimizer_attrs(attrs)
        assert dummy_optimizer_str == actual


def test_call_after_trial_of_base_sampler() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.CmaEsSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        independent_sampler, "after_trial", wraps=independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_is_compatible_search_space() -> None:
    transform = _SearchSpaceTransform(
        {
            "x0": optuna.distributions.UniformDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        }
    )

    assert optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x0": optuna.distributions.UniformDistribution(2, 3),
        },
    )

    # Same search space size, but different param names.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.UniformDistribution(2, 3),
            "foo": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )

    # x2 is added.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.UniformDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x2": optuna.distributions.DiscreteUniformDistribution(2, 3, q=0.1),
        },
    )

    # x0 is not found.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )
