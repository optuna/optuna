from typing import cast
from typing import Optional
from typing import Sequence
from typing import Tuple
from unittest.mock import patch
import warnings

import pytest
import torch

import optuna
from optuna import integration
from optuna.integration import BoTorchSampler
from optuna.samplers import RandomSampler
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial
from optuna.trial import Trial


@pytest.mark.parametrize("n_objectives", [1, 2, 4])
def test_botorch_candidates_func_none(n_objectives: int) -> None:
    n_trials = 3
    n_startup_trials = 2

    sampler = BoTorchSampler(n_startup_trials=n_startup_trials)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials

    # TODO(hvy): Do not check for the correct candidates function using private APIs.
    if n_objectives == 1:
        assert sampler._candidates_func is integration.botorch.qei_candidates_func
    elif n_objectives == 2:
        assert sampler._candidates_func is integration.botorch.qehvi_candidates_func
    elif n_objectives == 4:
        assert sampler._candidates_func is integration.botorch.qparego_candidates_func
    else:
        assert False, "Should not reach."


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

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=n_trials)

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

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(TypeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


def test_botorch_candidates_func_invalid_batch_size() -> None:
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(2, 1)  # Must have the batch size one, not two.

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


def test_botorch_candidates_func_invalid_dimensionality() -> None:
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(1, 1, 1)  # Must have one or two dimensions, not three.

    sampler = BoTorchSampler(candidates_func=candidates_func, n_startup_trials=1)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


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

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(ValueError):
        study.optimize(
            lambda t: sum(t.suggest_float(f"x{i}", 0, 1) for i in range(n_params)), n_trials=3
        )


@pytest.mark.parametrize("n_objectives", [1, 2, 4])
def test_botorch_constraints_func_none(n_objectives: int) -> None:
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        xs = sum(trial.params[f"x{i}"] for i in range(n_objectives))

        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (xs - 0.5,)

    n_trials = 4
    n_startup_trials = 2

    sampler = BoTorchSampler(constraints_func=constraints_func, n_startup_trials=n_startup_trials)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials


def test_botorch_constraints_func_invalid_type() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        x0 = trial.params["x0"]
        return x0 - 0.5  # Not a tuple, but it should be.

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(TypeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


def test_botorch_constraints_func_invalid_inconsistent_n_constraints() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        x0 = trial.params["x0"]
        return [x0 - 0.5] * trial.number  # Number of constraints may not change.

    sampler = BoTorchSampler(constraints_func=constraints_func, n_startup_trials=1)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(RuntimeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


def test_botorch_constraints_func_raises() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        if trial.number == 1:
            raise RuntimeError
        return (0.0,)

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(RuntimeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)

    assert len(study.trials) == 2

    for trial in study.trials:
        sys_con = trial.system_attrs["botorch:constraints"]

        expected_sys_con: Optional[Tuple[int]]

        if trial.number == 0:
            expected_sys_con = (0,)
        elif trial.number == 1:
            expected_sys_con = None
        else:
            assert False, "Should not reach."

        assert sys_con == expected_sys_con


def test_botorch_constraints_func_nan_warning() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        if trial.number == 1:
            raise RuntimeError
        return (0.0,)

    last_trial_number_candidates_func = None

    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        trial_number = train_x.size(0)

        assert train_con is not None

        if trial_number > 0:
            assert not train_con[0, :].isnan().any()
        if trial_number > 1:
            assert train_con[1, :].isnan().all()
        if trial_number > 2:
            assert not train_con[2, :].isnan().any()

        nonlocal last_trial_number_candidates_func
        last_trial_number_candidates_func = trial_number

        return torch.rand(1)

    sampler = BoTorchSampler(
        candidates_func=candidates_func,
        constraints_func=constraints_func,
        n_startup_trials=1,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(RuntimeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=None)

    assert len(study.trials) == 2

    # Warns when `train_con` contains NaN.
    with pytest.warns(UserWarning):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=2)

    assert len(study.trials) == 4

    assert last_trial_number_candidates_func == study.trials[-1].number


def test_botorch_constraints_func_none_warning() -> None:
    candidates_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        raise RuntimeError

    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        # `train_con` should be `None` if `constraints_func` always fails.
        assert train_con is None

        nonlocal candidates_func_call_count
        candidates_func_call_count += 1

        return torch.rand(1)

    sampler = BoTorchSampler(
        candidates_func=candidates_func,
        constraints_func=constraints_func,
        n_startup_trials=1,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)

    with pytest.raises(RuntimeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=None)

    assert len(study.trials) == 1

    # Warns when `train_con` becomes `None`.
    with pytest.warns(UserWarning), pytest.raises(RuntimeError):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=1)

    assert len(study.trials) == 2

    assert candidates_func_call_count == 1


def test_botorch_constraints_func_late() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        return (0,)

    last_trial_number_candidates_func = None

    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        trial_number = train_x.size(0)

        if trial_number < 3:
            assert train_con is None
        if trial_number == 3:
            assert train_con is not None
            assert train_con[:2, :].isnan().all()
            assert not train_con[2, :].isnan().any()

        nonlocal last_trial_number_candidates_func
        last_trial_number_candidates_func = trial_number

        return torch.rand(1)

    sampler = BoTorchSampler(
        candidates_func=candidates_func,
        n_startup_trials=1,
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=2)

    assert len(study.trials) == 2

    sampler = BoTorchSampler(
        candidates_func=candidates_func,
        constraints_func=constraints_func,
        n_startup_trials=1,
    )

    study.sampler = sampler

    # Warns when `train_con` contains NaN. Should not raise but will with NaN for previous trials
    # that were not computed with contraints.
    with pytest.warns(UserWarning):
        study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=2)

    assert len(study.trials) == 4

    assert last_trial_number_candidates_func == study.trials[-1].number


def test_botorch_n_startup_trials() -> None:
    independent_sampler = RandomSampler()
    sampler = BoTorchSampler(n_startup_trials=2, independent_sampler=independent_sampler)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)

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
    def objective(trial: Trial) -> float:
        x0 = trial.suggest_float("x0", 0, 1)
        x1 = trial.suggest_float("x1", 0.1, 1, log=True)
        x2 = trial.suggest_float("x2", 0, 1, step=0.1)
        x3 = trial.suggest_int("x3", 0, 2)
        x4 = trial.suggest_int("x4", 2, 4, log=True)
        x5 = trial.suggest_int("x5", 0, 4, step=2)
        x6 = cast(float, trial.suggest_categorical("x6", [0.1, 0.2, 0.3]))
        return x0 + x1 + x2 + x3 + x4 + x5 + x6

    sampler = BoTorchSampler()

    study = optuna.create_study(direction="minimize", sampler=sampler)
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

    study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage)
    study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)

    other_study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage)
    with pytest.raises(RuntimeError):
        other_study.optimize(lambda t: t.suggest_float("x0", 0, 1), n_trials=3)


def test_reseed_rng() -> None:
    independent_sampler = RandomSampler()
    sampler = BoTorchSampler(independent_sampler=independent_sampler)
    original_independent_sampler_seed = cast(RandomSampler, sampler._independent_sampler)._rng.seed

    sampler.reseed_rng()
    assert (
        original_independent_sampler_seed
        != cast(RandomSampler, sampler._independent_sampler)._rng.seed
    )


def test_call_after_trial_of_independent_sampler() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = BoTorchSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        independent_sampler, "after_trial", wraps=independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1
