from typing import cast
from typing import Optional
from typing import Sequence
from unittest.mock import patch

import pytest
import torch

from optuna import multi_objective
from optuna.integration import BoTorchSampler
from optuna.multi_objective.samplers._random import RandomMultiObjectiveSampler
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.storages import RDBStorage
from optuna.trial import Trial


def test_botorch_default_single_objective() -> None:
    sampler = BoTorchSampler()

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)

    assert len(study.trials) == 3


def test_botorch_default_multi_objective() -> None:
    sampler = BoTorchSampler()

    study = multi_objective.create_study(["minimize", "maximize"], sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)], n_trials=3
    )

    assert len(study.trials) == 3


def test_botorch_candidates_func() -> None:
    candidates_func_call_count = 0

    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        assert train_con is None

        candidates = torch.rand(1)

        nonlocal candidates_func_call_count
        candidates_func_call_count += 1

        return candidates

    n_trials = 3
    n_startup_trials = 1

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=n_startup_trials)

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert candidates_func_call_count == n_trials - n_startup_trials


def test_botorch_candidates_func_invalid_type() -> None:
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        # Must be a `torch.Tensor`, not a list.
        return torch.rand(1).tolist()  # type: ignore

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = multi_objective.create_study(["minimize"], sampler=sampler)

    with pytest.raises(TypeError):
        study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_botorch_candidates_func_invalid_batch_size() -> None:
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(2, 1)  # Must have the batch size one, not two.

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = multi_objective.create_study(["minimize"], sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_botorch_candidates_func_invalid_dimensionality() -> None:
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(1, 1, 1)  # Must have one or two dimensions, not three.

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = multi_objective.create_study(["minimize"], sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_botorch_candidates_func_invalid_candidates_size() -> None:
    n_params = 3

    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(n_params - 1)  # Must return candidates for all parameters.

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = multi_objective.create_study(["minimize"] * n_params, sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(
            lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_params)], n_trials=3
        )


def test_botorch_constraints_func_default_single_objective() -> None:
    constraints_func_call_count = 0

    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]

        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (x0 - 0.5,)

    n_trials = 4
    n_startup_trials = 2

    sampler = BoTorchSampler(constraints_func=constraints_func, n_startup_trials=n_startup_trials)

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=n_trials)

    assert len(study.trials) == n_trials
    # Only constraints up to the previous trial is computed.
    assert constraints_func_call_count == n_trials - 1


def test_botorch_constraints_func_default_multie_objective() -> None:
    constraints_func_call_count = 0

    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]
        x1 = trial.params["x1"]

        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (x0 + x1 - 0.5,)

    n_trials = 4
    n_startup_trials = 2

    sampler = BoTorchSampler(constraints_func=constraints_func, n_startup_trials=n_startup_trials)

    study = multi_objective.create_study(["minimize", "maximize"], sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials
    # Only constraints up to the previous trial is computed.
    assert constraints_func_call_count == n_trials - 1


def test_botorch_constraints_func_invalid_type() -> None:
    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]
        return x0 - 0.5  # Not a tuple, but it should be.

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = multi_objective.create_study(["minimize"], sampler=sampler)

    with pytest.raises(TypeError):
        study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_botorch_n_startup_trials() -> None:
    independent_sampler = RandomMultiObjectiveSampler()
    sampler = BoTorchSampler(n_startup_trials=2, independent_sampler=independent_sampler)
    study = multi_objective.create_study(["minimize", "maximize"], sampler=sampler)

    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_independent, patch.object(
        sampler, "sample_relative", wraps=sampler.sample_relative
    ) as mock_relative:
        study.optimize(
            lambda t: [t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)], n_trials=3
        )
        assert mock_independent.call_count == 4  # The objective function has two parameters.
        assert mock_relative.call_count == 3


def test_botorch_distributions() -> None:
    def objective(trial: Trial) -> Sequence[float]:
        x0 = trial.suggest_float("x0", 0, 1)
        x1 = trial.suggest_float("x1", 0.1, 1, log=True)
        x2 = trial.suggest_float("x2", 0, 1, step=0.1)
        x3 = trial.suggest_int("x3", 0, 2)
        x4 = trial.suggest_int("x4", 2, 4, log=True)
        x5 = trial.suggest_int("x5", 0, 4, step=2)
        x6 = cast(float, trial.suggest_categorical("x6", [0.1, 0.2, 0.3]))
        return [x0 + x1 + x2 + x3 + x4 + x5 + x6]

    sampler = BoTorchSampler()

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3


def test_botorch_invalid_different_studies() -> None:
    # Using the same sampler with different studies should yield an error since the sampler is
    # stateful holding the computed constraints. Two studies are considefered different if their
    # IDs differ.
    # We use the RDB storage since this check does not work for the in-memory storage where all
    # study IDs are identically 0.
    storage = RDBStorage("sqlite:///:memory:")

    sampler = BoTorchSampler()

    study = multi_objective.create_study(["minimize"], sampler=sampler, storage=storage)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)

    other_study = multi_objective.create_study(["minimize"], sampler=sampler, storage=storage)
    with pytest.raises(RuntimeError):
        other_study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_reseed_rng() -> None:
    independent_sampler = RandomMultiObjectiveSampler()
    sampler = BoTorchSampler(independent_sampler=independent_sampler)
    original_independent_sampler_seed = cast(
        RandomMultiObjectiveSampler, sampler._independent_sampler
    )._sampler._rng.seed

    sampler.reseed_rng()
    assert (
        original_independent_sampler_seed
        != cast(RandomMultiObjectiveSampler, sampler._independent_sampler)._sampler._rng.seed
    )
