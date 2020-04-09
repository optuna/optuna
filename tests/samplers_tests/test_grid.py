from collections import OrderedDict
import itertools

import numpy as np
import pytest

import optuna
from optuna import samplers

if optuna.type_checking.TYPE_CHECKING:
    from collections import ValuesView  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Mapping  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union  # NOQA

    from optuna.samplers.grid import GridValueType  # NOQA
    from optuna.trial import Trial  # NOQA


def _n_grids(search_space):
    # type: (Mapping[str, Sequence[Union[str, float, None]]]) -> int

    return int(np.prod([len(v) for v in search_space.values()]))


def test_grid_sampler_experimental_warning():
    # type: () -> None

    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.GridSampler({"some_param": [0, 1]})


def test_study_optimize_with_single_search_space():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_uniform("b", -0.1, 0.1)
        c = trial.suggest_categorical("c", ("x", "y"))
        d = trial.suggest_discrete_uniform("d", -5, 5, 1)
        e = trial.suggest_loguniform("e", 0.0001, 1)

        if c == "x":
            return a * d
        else:
            return b * e

    # Test that all combinations of the grid is sampled.
    search_space = {
        "b": np.arange(-0.1, 0.1, 0.05),
        "c": ["x", "y"],
        "d": [-0.5, 0.5],
        "e": [0.1],
        "a": list(range(0, 100, 20)),
    }
    n_grids = _n_grids(search_space)
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=n_grids)

    def sorted_values(d):
        # type: (Mapping[str, Sequence[GridValueType]]) -> ValuesView[Sequence[GridValueType]]

        return OrderedDict(sorted(d.items())).values()

    all_grids = itertools.product(*sorted_values(search_space))
    all_suggested_values = [tuple([p for p in sorted_values(t.params)]) for t in study.trials]
    assert set(all_grids) == set(all_suggested_values)

    ids = sorted([t.system_attrs["grid_id"] for t in study.trials])
    assert ids == list(range(n_grids))

    # Test that an optimization fails if the number of trials is more than that of all grids.
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1)

    # Test a non-existing parameter name in the grid.
    search_space = {"a": list(range(0, 100, 20))}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    with pytest.raises(ValueError):
        study.optimize(objective)

    # Test a value with out of range.
    search_space = {
        "a": [110],  # 110 is out of range specified by the suggest method.
        "b": [0],
        "c": ["x"],
        "d": [0],
        "e": [0.1],
    }
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    with pytest.raises(ValueError):
        study.optimize(objective)


def test_study_optimize_with_multiple_search_spaces():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_uniform("b", -100, 100)

        return a * b

    # Run 3 trials with a search space.
    search_space_0 = {"a": [0, 50], "b": [-50, 0, 50]}
    sampler_0 = samplers.GridSampler(search_space_0)
    study = optuna.create_study(sampler=sampler_0)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    for t in study.trials:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])

    # Run 2 trials with another space.
    search_space_1 = {"a": [0, 25], "b": [-50]}
    sampler_1 = samplers.GridSampler(search_space_1)
    study.sampler = sampler_1
    study.optimize(objective, n_trials=2)

    assert not sampler_0._same_search_space(sampler_1._search_space)
    assert len(study.trials) == 5
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])

    # Run 3 trials with the first search space again.
    study.sampler = sampler_0
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 8
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[5:]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])


def test_cast_value():
    # type: () -> None

    samplers.GridSampler._check_value("x", None)
    samplers.GridSampler._check_value("x", True)
    samplers.GridSampler._check_value("x", False)
    samplers.GridSampler._check_value("x", -1)
    samplers.GridSampler._check_value("x", -1.5)
    samplers.GridSampler._check_value("x", float("nan"))
    samplers.GridSampler._check_value("x", "foo")
    samplers.GridSampler._check_value("x", "")

    with pytest.raises(ValueError):
        samplers.GridSampler._check_value("x", [1])


def test_has_same_search_space():
    # type: () -> None

    search_space = {"x": [3, 2, 1], "y": ["a", "b", "c"]}  # type: Dict[str, List[Union[int, str]]]
    sampler = samplers.GridSampler(search_space)
    assert sampler._same_search_space(search_space)
    assert sampler._same_search_space({"x": np.array([3, 2, 1]), "y": ["a", "b", "c"]})
    assert sampler._same_search_space({"y": ["c", "a", "b"], "x": [1, 2, 3]})

    assert not sampler._same_search_space({"x": [3, 2, 1, 0], "y": ["a", "b", "c"]})
    assert not sampler._same_search_space({"x": [3, 2], "y": ["a", "b", "c"]})
