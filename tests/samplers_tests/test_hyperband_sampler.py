from __future__ import annotations

from functools import partial
import pickle
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import optuna
from optuna.distributions import FloatDistribution
from optuna.pruners import HyperbandPruner
from optuna.samplers import BaseSampler
from optuna.samplers import HyperbandSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_bracket_study() -> None:
    pruner = HyperbandPruner(max_resource=2, reduction_factor=2)
    assert len(pruner._pruners) == 2
    study = optuna.create_study(sampler=HyperbandSampler(RandomSampler), pruner=pruner)
    study.add_trials(
        [create_trial(value=0.0), create_trial(value=1.0), create_trial(state=TrialState.RUNNING)]
    )
    assert [trial.number for trial in study._get_trials(use_cache=True)] == [0, 1, 2]

    with patch.object(pruner, "_get_bracket_id", side_effect=lambda _, trial: trial.number % 2):
        bracket = pruner._create_bracket_study(study, 0)
        assert [trial.number for trial in bracket.get_trials()] == [0, 2]
        assert [trial.number for trial in bracket._get_trials(use_cache=True)] == [0, 2]
        assert [trial.number for trial in study._get_trials(use_cache=True)] == [0, 1, 2]
        assert [trial.number for trial in bracket.get_trials(states=(TrialState.COMPLETE,))] == [0]
        assert bracket.stop == study.stop

        running = bracket.trials[1]
        running.system_attrs["bracket"] = 0
        assert "bracket" not in study._storage.get_trial(running._trial_id).system_attrs
        bracket._storage.set_trial_system_attr(running._trial_id, "bracket", 0)
        assert study._storage.get_trial(running._trial_id).system_attrs["bracket"] == 0


def test_hyperband_sampler_reuses_one_child_for_each_bracket() -> None:
    children = [Mock(spec=BaseSampler), Mock(spec=BaseSampler)]
    for child in children:
        child.infer_relative_search_space.return_value = {}
        child.sample_relative.return_value = {}
        child.sample_independent.return_value = 0.5
    make_sampler = Mock(side_effect=children)
    pruner = HyperbandPruner(max_resource=2, reduction_factor=2)
    sampler = HyperbandSampler(make_sampler, seed=0)
    study = optuna.create_study(sampler=sampler, pruner=pruner)

    with patch.object(pruner, "_get_bracket_id", side_effect=lambda _, trial: trial.number % 2):
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)

    hooks = [
        "before_trial",
        "infer_relative_search_space",
        "sample_relative",
        "sample_independent",
        "after_trial",
    ]
    for child, trial_numbers in zip(children, ([0, 2], [1])):
        assert [(call[0], call.args[1].number) for call in child.mock_calls] == [
            (hook, number) for number in trial_numbers for hook in hooks
        ]
        assert all(call.args[0].sampler is child for call in child.mock_calls)
    assert make_sampler.call_count == 2
    assert len({call.kwargs["seed"] for call in make_sampler.call_args_list}) == 2


def test_hyperband_sampler_tpe_uses_bracket_local_search_space() -> None:
    pruner = HyperbandPruner(max_resource=2, reduction_factor=2)
    make_sampler = partial(TPESampler, n_startup_trials=1, multivariate=True)
    study = optuna.create_study(sampler=HyperbandSampler(make_sampler, seed=0), pruner=pruner)
    distribution = FloatDistribution(0.0, 1.0)
    study.add_trials(
        optuna.trial.create_trial(
            params={name: 0.5}, distributions={name: distribution}, value=0.5
        )
        for name in ("x", "y")
    )

    with patch.object(pruner, "_get_bracket_id", side_effect=lambda _, trial: trial.number % 2):
        for name in ("x", "y"):
            trial = study.ask()
            value = trial.suggest_float(name, 0.0, 1.0)
            assert trial.relative_search_space == {name: distribution}
            study.tell(trial, value)


def test_hyperband_sampler_reuses_history_until_after_trial() -> None:
    study = optuna.create_study(
        sampler=HyperbandSampler(seed=0), pruner=HyperbandPruner(max_resource=1)
    )

    def check_history(_: TPESampler, bracket: Study, trial: FrozenTrial, *args: Any) -> None:
        history = bracket._get_trials(deepcopy=False, use_cache=True)
        assert history[-1].params == trial.params

    with patch.object(TPESampler, "after_trial", autospec=True, side_effect=check_history):
        for _ in range(2):
            trial = study.ask()
            with patch.object(
                study._storage, "get_all_trials", wraps=study._storage.get_all_trials
            ) as get_all_trials:
                for name in ("x", "y", "z"):
                    trial.suggest_float(name, 0.0, 1.0)
                assert get_all_trials.call_count == 1
            study.tell(trial, 0.0)


def test_hyperband_sampler_preserves_bracket_pruning() -> None:
    pruner = HyperbandPruner(max_resource=2, reduction_factor=2)
    study = optuna.create_study(sampler=HyperbandSampler(RandomSampler, seed=0), pruner=pruner)

    # The best value belongs to bracket 1. Within bracket 0, only the worse trial is pruned.
    with patch.object(
        pruner, "_get_bracket_id", side_effect=lambda _, trial: int(trial.number == 0)
    ):
        for value, step, expected_pruned in [(0.0, 2, False), (1.0, 1, False), (2.0, 1, True)]:
            trial = study.ask()
            trial.report(value, step)
            assert trial.should_prune() is expected_pruned
            if expected_pruned:
                study.tell(trial, state=TrialState.PRUNED)
            else:
                study.tell(trial, value)


def test_hyperband_sampler_accepts_another_sampler_class_and_reproduces_results() -> None:
    def run() -> list[dict[str, Any]]:
        study = optuna.create_study(
            study_name="hyperband-random",
            sampler=HyperbandSampler(RandomSampler, seed=0),
            pruner=HyperbandPruner(max_resource=3),
        )
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=12)
        return [trial.params for trial in study.trials]

    assert run() == run()


@pytest.mark.parametrize(
    "pruner, message",
    [
        (optuna.pruners.NopPruner(), "requires HyperbandPruner"),
        (HyperbandPruner(), "requires an explicit max_resource"),
    ],
)
def test_hyperband_sampler_requires_a_pruner_with_fixed_resources(
    pruner: optuna.pruners.BasePruner, message: str
) -> None:
    study = optuna.create_study(sampler=HyperbandSampler(TPESampler), pruner=pruner)
    with pytest.raises(ValueError, match=message):
        study.ask()


def test_hyperband_pruner_warns_without_hyperband_sampler() -> None:
    study = optuna.create_study(pruner=HyperbandPruner(max_resource=2))
    trial = study.ask()
    trial.report(1.0, step=1)

    with pytest.warns(UserWarning, match="used without HyperbandSampler"):
        assert not trial.should_prune()


@pytest.mark.filterwarnings("ignore:HyperbandPruner is being used without HyperbandSampler")
def test_hyperband_pruner_initializes_auto_from_completed_trial() -> None:
    pruner = HyperbandPruner()
    study = optuna.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1.0, step=8)
    assert not trial.should_prune()
    assert not pruner._pruners
    study.tell(trial, 1.0)

    trial = study.ask()
    trial.report(1.0, step=0)
    assert not trial.should_prune()
    assert pruner._max_resource == 9
    assert len(pruner._pruners) == 3


@pytest.mark.parametrize(
    "max_resource, bootstrap_count, message",
    [
        ("not_appropriate", 0, "should be integer or 'auto'"),
        ("auto", 1, "are mutually incompatible"),
    ],
    ids=["invalid-max-resource", "auto-with-bootstrap"],
)
def test_hyperband_pruner_rejects_invalid_resources(
    max_resource: str, bootstrap_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HyperbandPruner(max_resource=max_resource, bootstrap_count=bootstrap_count)


def test_hyperband_sampler_reuses_child_after_restoring_study() -> None:
    study = optuna.create_study(
        sampler=HyperbandSampler(seed=0),
        pruner=HyperbandPruner(max_resource=3),
    )
    trial = study.ask()
    value = trial.suggest_float("x", 0.0, 1.0)
    restored: Study = pickle.loads(pickle.dumps(study))
    assert isinstance(restored.sampler, HyperbandSampler)

    with patch.object(
        restored.sampler, "_sampler_factory", side_effect=AssertionError("already created")
    ):
        assert isinstance(
            restored.sampler._bracket_study(restored, restored.trials[0]).sampler, TPESampler
        )
        assert restored.tell(trial.number, value).value == value
