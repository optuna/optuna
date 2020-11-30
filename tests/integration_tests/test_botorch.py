from typing import cast
from typing import Optional
from typing import Sequence
from unittest.mock import patch

from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
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
    def candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: Optional[torch.Tensor],
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        assert train_con is None

        model = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acqf = UpperConfidenceBound(model, beta=0.1)
        candidates, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=40)

        return candidates

    sampler = BoTorchSampler(candidates_func=candidates_func)

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)

    assert len(study.trials) == 3


def test_botorch_constraints_func_default_single_objective() -> None:
    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]
        return (x0 - 0.5,)

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = multi_objective.create_study(["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)

    assert len(study.trials) == 3


def test_botorch_constraints_func_default_single_objective_invalid_type() -> None:
    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]
        return x0 - 0.5  # Not a tuple, but it should be.

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = multi_objective.create_study(["minimize"], sampler=sampler)

    with pytest.raises(TypeError):
        study.optimize(lambda t: [t.suggest_float("x0", 0, 1)], n_trials=3)


def test_botorch_constraints_func_default_multie_objective() -> None:
    def constraints_func(
        study: MultiObjectiveStudy, trial: FrozenMultiObjectiveTrial
    ) -> Sequence[float]:
        x0 = trial.params["x0"]
        x1 = trial.params["x1"]
        return (x0 + x1 - 0.5,)

    sampler = BoTorchSampler(constraints_func=constraints_func)

    study = multi_objective.create_study(["minimize", "maximize"], sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)], n_trials=3
    )

    assert len(study.trials) == 3


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
