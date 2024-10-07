from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import _pytest.capture
from cmaes import CMA
from cmaes import CMAwM
from cmaes import SepCMA
import numpy as np
import pytest

import optuna
from optuna import create_trial
from optuna._transform import _SearchSpaceTransform
from optuna.testing.storages import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_consider_pruned_trials_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.CmaEsSampler(consider_pruned_trials=True)


def test_with_margin_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.CmaEsSampler(with_margin=True)


def test_lr_adapt_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.CmaEsSampler(lr_adapt=True)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize(
    "use_separable_cma, cma_class_str",
    [(False, "optuna.samplers._cmaes.cmaes.CMA"), (True, "optuna.samplers._cmaes.cmaes.SepCMA")],
)
@pytest.mark.parametrize("popsize", [None, 8])
def test_init_cmaes_opts(use_separable_cma: bool, cma_class_str: str, popsize: int | None) -> None:
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        use_separable_cma=use_separable_cma,
        popsize=popsize,
    )
    study = optuna.create_study(sampler=sampler)

    with patch(cma_class_str) as cma_class:
        cma_obj = MagicMock()
        cma_obj.ask.return_value = np.array((-1, -1))
        cma_obj.generation = 0
        cma_class.return_value = cma_obj
        study.optimize(
            lambda t: t.suggest_float("x", -1, 1) + t.suggest_float("y", -1, 1), n_trials=2
        )

        assert cma_class.call_count == 1

        _, actual_kwargs = cma_class.call_args
        assert np.array_equal(actual_kwargs["mean"], np.array([0.5, 0.5]))
        assert actual_kwargs["sigma"] == 0.1
        assert np.allclose(actual_kwargs["bounds"], np.array([(0, 1), (0, 1)]))
        assert actual_kwargs["seed"] == np.random.RandomState(1).randint(1, np.iinfo(np.int32).max)
        assert actual_kwargs["n_max_resampling"] == 10 * 2
        expected_popsize = 4 + math.floor(3 * math.log(2)) if popsize is None else popsize
        assert actual_kwargs["population_size"] == expected_popsize


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("popsize", [None, 8])
def test_init_cmaes_opts_with_margin(popsize: int | None) -> None:
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        popsize=popsize,
        with_margin=True,
    )
    study = optuna.create_study(sampler=sampler)

    with patch("optuna.samplers._cmaes.cmaes.CMAwM") as cma_class:
        cma_obj = MagicMock()
        cma_obj.ask.return_value = np.array((-1, -1))
        cma_obj.generation = 0
        cma_class.return_value = cma_obj
        study.optimize(
            lambda t: t.suggest_float("x", -1, 1) + t.suggest_int("y", -1, 1), n_trials=2
        )

        assert cma_class.call_count == 1

        _, actual_kwargs = cma_class.call_args
        assert np.array_equal(actual_kwargs["mean"], np.array([0.5, 0.5]))
        assert actual_kwargs["sigma"] == 0.1
        assert np.allclose(actual_kwargs["bounds"], np.array([(0, 1), (0, 1)]))
        assert np.allclose(actual_kwargs["steps"], np.array([0.0, 0.5]))
        assert actual_kwargs["seed"] == np.random.RandomState(1).randint(1, np.iinfo(np.int32).max)
        assert actual_kwargs["n_max_resampling"] == 10 * 2
        expected_popsize = 4 + math.floor(3 * math.log(2)) if popsize is None else popsize
        assert actual_kwargs["population_size"] == expected_popsize


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("popsize", [None, 8])
def test_init_cmaes_opts_lr_adapt(popsize: int | None) -> None:
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        popsize=popsize,
        lr_adapt=True,
    )
    study = optuna.create_study(sampler=sampler)

    with patch("optuna.samplers._cmaes.cmaes.CMA") as cma_class:
        cma_obj = MagicMock()
        cma_obj.ask.return_value = np.array((-1, -1))
        cma_obj.generation = 0
        cma_class.return_value = cma_obj
        study.optimize(
            lambda t: t.suggest_float("x", -1, 1) + t.suggest_float("y", -1, 1), n_trials=2
        )

        assert cma_class.call_count == 1

        _, actual_kwargs = cma_class.call_args
        assert actual_kwargs["lr_adapt"] is True


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("with_margin", [False, True])
def test_warm_starting_cmaes(with_margin: bool) -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y

    source_study = optuna.create_study()
    source_study.optimize(objective, 20)
    source_trials = source_study.get_trials(deepcopy=False)

    with patch("optuna.samplers._cmaes.cmaes.get_warm_start_mgd") as mock_func_ws:
        mock_func_ws.return_value = (np.zeros(2), 0.0, np.zeros((2, 2)))
        sampler = optuna.samplers.CmaEsSampler(
            seed=1, n_startup_trials=1, with_margin=with_margin, source_trials=source_trials
        )
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, 2)
        assert mock_func_ws.call_count == 1


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("with_margin", [False, True])
def test_warm_starting_cmaes_maximize(with_margin: bool) -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        # Objective values are negative.
        return -(x**2) - (y - 5) ** 2

    source_study = optuna.create_study(direction="maximize")
    source_study.optimize(objective, 20)
    source_trials = source_study.get_trials(deepcopy=False)

    with patch("optuna.samplers._cmaes.cmaes.get_warm_start_mgd") as mock_func_ws:
        mock_func_ws.return_value = (np.zeros(2), 0.0, np.zeros((2, 2)))
        sampler = optuna.samplers.CmaEsSampler(
            seed=1, n_startup_trials=1, with_margin=with_margin, source_trials=source_trials
        )
        study = optuna.create_study(sampler=sampler, direction="maximize")
        study.optimize(objective, 2)
        assert mock_func_ws.call_count == 1

        solutions_arg = mock_func_ws.call_args[0][0]
        is_positive = [x[1] >= 0 for x in solutions_arg]
        assert all(is_positive)


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

    with pytest.raises(ValueError):
        optuna.samplers.CmaEsSampler(use_separable_cma=True, with_margin=True)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("with_margin", [False, True])
def test_incompatible_search_space(with_margin: bool) -> None:
    def objective1(trial: optuna.Trial) -> float:
        x0 = trial.suggest_float("x0", 2, 3)
        x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
        return x0 + x1

    source_study = optuna.create_study()
    source_study.optimize(objective1, 20)

    # Should not raise an exception.
    sampler = optuna.samplers.CmaEsSampler(
        with_margin=with_margin, source_trials=source_study.trials
    )
    target_study1 = optuna.create_study(sampler=sampler)
    target_study1.optimize(objective1, 20)

    def objective2(trial: optuna.Trial) -> float:
        x0 = trial.suggest_float("x0", 2, 3)
        x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
        x2 = trial.suggest_float("x2", 1e-2, 1e2, log=True)
        return x0 + x1 + x2

    # Should raise an exception.
    sampler = optuna.samplers.CmaEsSampler(
        with_margin=with_margin, source_trials=source_study.trials
    )
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
    independent_sampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.CmaEsSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)

    # If search space is one dimensional, the independent sampler is always used.
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_object:
        study.optimize(lambda t: t.suggest_int("x", -1, 1), n_trials=2)
        assert mock_object.call_count == 2


def test_sample_relative_n_startup_trials() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
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


def test_get_trials() -> None:
    with patch(
        "optuna.Study._get_trials",
        new=Mock(side_effect=lambda deepcopy, use_cache: _create_trials()),
    ):
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


def _create_trials() -> list[FrozenTrial]:
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


@pytest.mark.parametrize(
    "options, key",
    [
        ({"with_margin": False, "use_separable_cma": False}, "cma:"),
        ({"with_margin": True, "use_separable_cma": False}, "cmawm:"),
        ({"with_margin": False, "use_separable_cma": True}, "sepcma:"),
    ],
)
def test_sampler_attr_key(options: dict[str, bool], key: str) -> None:
    # Test sampler attr_key property.
    sampler = optuna.samplers.CmaEsSampler(
        with_margin=options["with_margin"], use_separable_cma=options["use_separable_cma"]
    )
    assert sampler._attr_keys.optimizer(0).startswith(key)
    assert sampler._attr_keys.popsize().startswith(key)
    assert sampler._attr_keys.n_restarts().startswith(key)
    assert sampler._attr_keys.n_restarts_with_large.startswith(key)
    assert sampler._attr_keys.poptype.startswith(key)
    assert sampler._attr_keys.small_n_eval.startswith(key)
    assert sampler._attr_keys.large_n_eval.startswith(key)
    assert sampler._attr_keys.generation(0).startswith(key)

    for restart_strategy in ["ipop", "bipop"]:
        sampler._restart_strategy = restart_strategy
        for i in range(3):
            assert sampler._attr_keys.generation(i).startswith(
                (key + "{}:restart_{}:".format(restart_strategy, i) + "generation")
            )


@pytest.mark.parametrize("popsize", [None, 16])
def test_population_size_is_multiplied_when_enable_ipop(popsize: int | None) -> None:
    inc_popsize = 2
    sampler = optuna.samplers.CmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        restart_strategy="ipop",
        popsize=popsize,
        inc_popsize=inc_popsize,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        _ = trial.suggest_float("x", -1, 1)
        _ = trial.suggest_float("y", -1, 1)
        return 1.0

    with patch("optuna.samplers._cmaes.cmaes.CMA") as cma_class_mock, patch(
        "optuna.samplers._cmaes.pickle"
    ) as pickle_mock:
        pickle_mock.dump.return_value = b"serialized object"

        should_stop_mock = MagicMock()
        should_stop_mock.return_value = True

        cma_obj = CMA(
            mean=np.array([-1, -1], dtype=float),
            sigma=1.3,
            bounds=np.array([[-1, 1], [-1, 1]], dtype=float),
            population_size=popsize,  # Already tested by test_init_cmaes_opts().
        )
        cma_obj.should_stop = should_stop_mock
        cma_class_mock.return_value = cma_obj

        initial_popsize = cma_obj.population_size
        study.optimize(objective, n_trials=2 + initial_popsize)
        assert cma_obj.should_stop.call_count == 1

        _, actual_kwargs = cma_class_mock.call_args
        assert actual_kwargs["population_size"] == inc_popsize * initial_popsize


@pytest.mark.parametrize("sampler_opts", [{}, {"use_separable_cma": True}, {"with_margin": True}])
def test_restore_optimizer_from_substrings(sampler_opts: dict[str, Any]) -> None:
    popsize = 8
    sampler = optuna.samplers.CmaEsSampler(popsize=popsize, **sampler_opts)
    optimizer = sampler._restore_optimizer([])
    assert optimizer is None

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=popsize + 2)
    optimizer = sampler._restore_optimizer(study.trials)

    assert optimizer is not None
    assert optimizer.generation == 1
    if sampler._with_margin:
        assert isinstance(optimizer, CMAwM)
    elif sampler._use_separable_cma:
        assert isinstance(optimizer, SepCMA)
    else:
        assert isinstance(optimizer, CMA)


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
        {"restart_strategy": "ipop", "use_separable_cma": True},
        {"restart_strategy": "bipop", "use_separable_cma": True},
        {"restart_strategy": "ipop", "with_margin": True},
        {"restart_strategy": "bipop", "with_margin": True},
    ],
)
def test_restore_optimizer_after_restart(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    if sampler_opts.get("with_margin"):
        cma_class = CMAwM
    elif sampler_opts.get("use_separable_cma"):
        cma_class = SepCMA
    else:
        cma_class = CMA
    with patch.object(cma_class, "should_stop") as mock_method:
        mock_method.return_value = True
        sampler = optuna.samplers.CmaEsSampler(popsize=5, **sampler_opts)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=5 + 2)

    optimizer = sampler._restore_optimizer(study.trials, 1)
    assert optimizer is not None
    assert optimizer.generation == 0


@pytest.mark.parametrize(
    "sampler_opts, restart_strategy",
    [
        ({"use_separable_cma": True}, "ipop"),
        ({"use_separable_cma": True}, "bipop"),
        ({"with_margin": True}, "ipop"),
        ({"with_margin": True}, "bipop"),
    ],
)
def test_restore_optimizer_with_other_option(
    sampler_opts: dict[str, Any], restart_strategy: str
) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    with patch.object(CMA, "should_stop") as mock_method:
        mock_method.return_value = True
        sampler = optuna.samplers.CmaEsSampler(popsize=5, restart_strategy=restart_strategy)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=5 + 2)

    # Restore optimizer via SepCMA or CMAwM samplers.
    sampler = optuna.samplers.CmaEsSampler(**sampler_opts)
    optimizer = sampler._restore_optimizer(study.trials)
    assert optimizer is None


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
        {"restart_strategy": "ipop", "use_separable_cma": True},
        {"restart_strategy": "bipop", "use_separable_cma": True},
        {"restart_strategy": "ipop", "with_margin": True},
        {"restart_strategy": "bipop", "with_margin": True},
    ],
)
def test_get_solution_trials(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    popsize = 5
    sampler = optuna.samplers.CmaEsSampler(popsize=popsize, **sampler_opts)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=popsize + 2)

    # The number of solutions for generation 0 equals population size.
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == popsize

    # The number of solutions for generation 1 is 1.
    assert len(sampler._get_solution_trials(study.trials, 1, 0)) == 1


@pytest.mark.parametrize(
    "sampler_opts, restart_strategy",
    [
        ({"use_separable_cma": True}, "ipop"),
        ({"use_separable_cma": True}, "bipop"),
        ({"with_margin": True}, "ipop"),
        ({"with_margin": True}, "bipop"),
    ],
)
def test_get_solution_trials_with_other_options(
    sampler_opts: dict[str, Any], restart_strategy: str
) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    sampler = optuna.samplers.CmaEsSampler(popsize=5, restart_strategy=restart_strategy)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5 + 2)

    # The number of solutions is 0 after changed samplers
    sampler = optuna.samplers.CmaEsSampler(**sampler_opts)
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == 0


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
        {"restart_strategy": "ipop", "use_separable_cma": True},
        {"restart_strategy": "bipop", "use_separable_cma": True},
        {"restart_strategy": "ipop", "with_margin": True},
        {"restart_strategy": "bipop", "with_margin": True},
    ],
)
def test_get_solution_trials_after_restart(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    if sampler_opts.get("with_margin"):
        cma_class = CMAwM
    elif sampler_opts.get("use_separable_cma"):
        cma_class = SepCMA
    else:
        cma_class = CMA

    popsize = 5
    with patch.object(cma_class, "should_stop") as mock_method:
        mock_method.return_value = True
        sampler = optuna.samplers.CmaEsSampler(popsize=popsize, **sampler_opts)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=popsize + 2)

    # The number of solutions for generation=0 and n_restarts=0 equals population size.
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == popsize

    # The number of solutions for generation=1 and n_restarts=0 is 0.
    assert len(sampler._get_solution_trials(study.trials, 1, 0)) == 0

    # The number of solutions for generation=0 and n_restarts=1 is 1 since it was restarted.
    assert len(sampler._get_solution_trials(study.trials, 0, 1)) == 1


@pytest.mark.parametrize(
    "dummy_optimizer_str,attr_len",
    [
        ("012", 1),
        ("01234", 1),
        ("012345", 2),
    ],
)
def test_split_and_concat_optimizer_string(dummy_optimizer_str: str, attr_len: int) -> None:
    sampler = optuna.samplers.CmaEsSampler()
    with patch("optuna.samplers._cmaes._SYSTEM_ATTR_MAX_LENGTH", 5):
        attrs = sampler._split_optimizer_str(dummy_optimizer_str)
        assert len(attrs) == attr_len
        actual = sampler._concat_optimizer_attrs(attrs)
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
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        }
    )

    assert optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x0": optuna.distributions.FloatDistribution(2, 3),
        },
    )

    # Same search space size, but different param names.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "foo": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )

    # x2 is added.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x2": optuna.distributions.FloatDistribution(2, 3, step=0.1),
        },
    )

    # x0 is not found.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )


def test_internal_optimizer_with_margin() -> None:
    def objective_discrete(trial: optuna.Trial) -> float:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y

    def objective_mixed(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y

    def objective_continuous(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y

    objectives = [objective_discrete, objective_mixed, objective_continuous]
    for objective in objectives:
        with patch("optuna.samplers._cmaes.cmaes.CMAwM") as cmawm_class_mock:
            sampler = optuna.samplers.CmaEsSampler(with_margin=True)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=2)
            assert cmawm_class_mock.call_count == 1


@pytest.mark.parametrize("warn_independent_sampling", [True, False])
def test_warn_independent_sampling(
    capsys: _pytest.capture.CaptureFixture, warn_independent_sampling: bool
) -> None:
    def objective_single(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("x", 0, 1)

    def objective_shrink(trial: optuna.trial.Trial) -> float:
        if trial.number != 5:
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            z = trial.suggest_float("z", 0, 1)
            return x + y + z
        else:
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            return x + y

    def objective_expand(trial: optuna.trial.Trial) -> float:
        if trial.number != 5:
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            return x + y
        else:
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            z = trial.suggest_float("z", 0, 1)
            return x + y + z

    for objective in [objective_single, objective_shrink, objective_expand]:
        # We need to reconstruct our default handler to properly capture stderr.
        optuna.logging._reset_library_root_logger()
        optuna.logging.enable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.CmaEsSampler(warn_independent_sampling=warn_independent_sampling)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=10)

        _, err = capsys.readouterr()
        assert (err != "") == warn_independent_sampling


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("with_margin", [False, True])
@pytest.mark.parametrize("storage_name", ["sqlite", "journal"])
def test_rdb_storage(with_margin: bool, storage_name: str) -> None:
    # Confirm `study._storage.set_trial_system_attr` does not fail in several storages.
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y

    with StorageSupplier(storage_name) as storage:
        study = optuna.create_study(
            sampler=optuna.samplers.CmaEsSampler(with_margin=with_margin),
            storage=storage,
        )
        study.optimize(objective, n_trials=3)
