import datetime
import os
from typing import Optional

import pytest

import optuna
from optuna._imports import try_import
from optuna.integration import TorchDistributedTrial
from optuna.testing.pruners import DeterministicPruner
from optuna.testing.storages import StorageSupplier


with try_import():
    import torch
    import torch.distributed as dist

pytestmark = pytest.mark.integration

STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "cached_sqlite",
    "journal",
    "journal_redis",
]


@pytest.fixture(scope="session", autouse=True)
def init_process_group() -> None:
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        pytest.skip("This test is expected to be launch with mpirun.")

    # This function is automatically called at the beginning of the pytest session.
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "20000"

    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=15))


def test_torch_distributed_trial_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        if dist.get_rank() == 0:
            study = optuna.create_study()
            TorchDistributedTrial(study.ask())
        else:
            TorchDistributedTrial(None)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_torch_distributed_trial_invalid_argument() -> None:
    with pytest.raises(ValueError):
        if dist.get_rank() == 0:
            TorchDistributedTrial(None)
        else:
            study = optuna.create_study()
            TorchDistributedTrial(study.ask())


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_float(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_float("x", 0, 1)
        assert 0 <= x1 <= 1

        x2 = trial.suggest_float("x", 0, 1)
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_uniform(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_uniform("x", 0, 1)
        assert 0 <= x1 <= 1

        x2 = trial.suggest_uniform("x", 0, 1)
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_loguniform(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_loguniform("x", 1e-7, 1)
        assert 1e-7 <= x1 <= 1

        x2 = trial.suggest_loguniform("x", 1e-7, 1)
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_discrete_uniform(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_discrete_uniform("x", 0, 10, 2)
        assert 0 <= x1 <= 10
        assert x1 % 2 == 0

        x2 = trial.suggest_discrete_uniform("x", 0, 10, 2)
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_int(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_int("x", 0, 10)
        assert 0 <= x1 <= 10

        x2 = trial.suggest_int("x", 0, 10)
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_categorical(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        x1 = trial.suggest_categorical("x", ("a", "b", "c"))
        assert x1 in {"a", "b", "c"}

        x2 = trial.suggest_categorical("x", ("a", "b", "c"))
        assert x1 == x2


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_report(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study: Optional[optuna.study.Study] = None
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.report(1, 0)

        if dist.get_rank() == 0:
            assert study is not None
            study.trials[0].intermediate_values[0] == 1


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_report_nan(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study: Optional[optuna.study.Study] = None
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        with pytest.raises(TypeError):
            trial.report("abc", 0)  # type: ignore[arg-type]

        if dist.get_rank() == 0:
            assert study is not None
            assert len(study.trials[0].intermediate_values) == 0


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("is_pruning", [False, True])
def test_should_prune(storage_mode: str, is_pruning: bool) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage, pruner=DeterministicPruner(is_pruning))
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.report(1, 0)
        assert trial.should_prune() == is_pruning


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_user_attrs(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.set_user_attr("dataset", "mnist")
        trial.set_user_attr("batch_size", 128)

        assert trial.user_attrs["dataset"] == "mnist"
        assert trial.user_attrs["batch_size"] == 128


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_user_attrs_with_exception() -> None:
    with StorageSupplier("sqlite") as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        with pytest.raises(TypeError):
            trial.set_user_attr("not serializable", torch.Tensor([1, 2]))


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_number(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        assert trial.number == 0


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_datetime_start(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        assert isinstance(trial.datetime_start, datetime.datetime)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_params(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.suggest_float("f", 0, 1)
        trial.suggest_int("i", 0, 1)
        trial.suggest_categorical("c", ("a", "b", "c"))

        params = trial.params
        assert 0 <= params["f"] <= 1
        assert 0 <= params["i"] <= 1
        assert params["c"] in {"a", "b", "c"}


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_distributions(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.suggest_float("u", 0, 1)
        trial.suggest_float("lu", 1e-7, 1, log=True)
        trial.suggest_float("du", 0, 1, step=0.5)
        trial.suggest_int("i", 0, 1)
        trial.suggest_int("il", 1, 128, log=True)
        trial.suggest_categorical("c", ("a", "b", "c"))

        distributions = trial.distributions
        assert distributions["u"] == optuna.distributions.FloatDistribution(0, 1)
        assert distributions["lu"] == optuna.distributions.FloatDistribution(1e-7, 1, log=True)
        assert distributions["du"] == optuna.distributions.FloatDistribution(0, 1, step=0.5)
        assert distributions["i"] == optuna.distributions.IntDistribution(0, 1)
        assert distributions["il"] == optuna.distributions.IntDistribution(1, 128, log=True)
        assert distributions["c"] == optuna.distributions.CategoricalDistribution(("a", "b", "c"))


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_updates_properties(storage_mode: str) -> None:
    """Check for any distributed deadlock following a property read."""
    with StorageSupplier(storage_mode) as storage:
        if dist.get_rank() == 0:
            study = optuna.create_study(storage=storage)
            trial = TorchDistributedTrial(study.ask())
        else:
            trial = TorchDistributedTrial(None)

        trial.suggest_float("f", 0, 1)
        trial.suggest_int("i", 0, 1)
        trial.suggest_categorical("c", ("a", "b", "c"))

        property_names = [
            p
            for p in dir(TorchDistributedTrial)
            if isinstance(getattr(TorchDistributedTrial, p), property)
        ]

        # Rank 0 can read properties without deadlock.
        if dist.get_rank() == 0:
            [getattr(trial, p) for p in property_names]

        dist.barrier()

        # Same with rank 1.
        if dist.get_rank() == 1:
            [getattr(trial, p) for p in property_names]

        dist.barrier()


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_pass_frozen_trial_to_torch_distributed(storage_mode: str) -> None:
    # Regression test of #4697
    def objective(trial: optuna.trial.BaseTrial) -> float:
        trial = optuna.integration.TorchDistributedTrial(trial if dist.get_rank() == 0 else None)
        x = trial.suggest_float("x", low=-100, high=100)
        return x * x

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(direction="minimize", storage=storage)
        study.optimize(objective, n_trials=1)
        best_trial = study.best_trial

        objective(best_trial)
