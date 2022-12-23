import json
import os
import tempfile
from typing import Dict
from typing import Type
from typing import Union
from unittest import mock

import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.allennlp import AllenNLPPruningCallback
from optuna.integration.allennlp._pruner import _create_pruner
from optuna.integration.allennlp._variables import _VariableManager
from optuna.testing.pruners import DeterministicPruner


with try_import():
    import _jsonnet
    import allennlp.data
    import allennlp.data.dataset_readers
    import allennlp.data.tokenizers
    import allennlp.models
    import allennlp.modules
    import allennlp.modules.seq2vec_encoders
    import allennlp.modules.text_field_embedders
    import allennlp.training
    import psutil
    import torch.optim

pytestmark = pytest.mark.integration


def test_build_params() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    executor = optuna.integration.AllenNLPExecutor(
        trial, "tests/integration_tests/allennlp_tests/test.jsonnet", "test"
    )
    params = executor._build_params()

    assert params["model"]["dropout"] == 0.1
    assert params["model"]["input_size"] == 100
    assert params["model"]["hidden_size"] == [100, 200, 300]


def test_build_params_overwriting_environment_variable() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    os.environ["TRAIN_PATH"] = "tests/integration_tests/allennlp_tests/sentences.train"
    os.environ["VALID_PATH"] = "tests/integration_tests/allennlp_tests/sentences.valid"
    executor = optuna.integration.AllenNLPExecutor(
        trial,
        "tests/integration_tests/allennlp_tests/example_with_environment_variables.jsonnet",
        "test",
    )
    params = executor._build_params()
    os.environ.pop("TRAIN_PATH")
    os.environ.pop("VALID_PATH")
    assert params["train_data_path"] == "tests/integration_tests/allennlp_tests/sentences.train"
    assert (
        params["validation_data_path"] == "tests/integration_tests/allennlp_tests/sentences.valid"
    )


def test_build_params_when_optuna_and_environment_variable_both_exist() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("LEARNING_RATE", 1e-2, 1e-2)
    os.environ["TRAIN_PATH"] = "tests/integration_tests/allennlp_tests/sentences.train"
    os.environ["VALID_PATH"] = "tests/integration_tests/allennlp_tests/sentences.valid"
    os.environ["LEARNING_RATE"] = "1e-3"
    os.environ["DROPOUT"] = "0.0"
    executor = optuna.integration.AllenNLPExecutor(
        trial,
        "tests/integration_tests/allennlp_tests/example_with_environment_variables.jsonnet",
        "test",
    )
    params = executor._build_params()
    os.environ.pop("TRAIN_PATH")
    os.environ.pop("VALID_PATH")
    os.environ.pop("LEARNING_RATE")
    os.environ.pop("DROPOUT")

    # Optuna trial overwrites a parameter specified by environment variable.
    assert params["trainer"]["optimizer"]["lr"] == 1e-2
    path = params["model"]["text_field_embedder"]["token_embedders"]["token_characters"]["dropout"]
    assert path == 0.0


def test_missing_config_file() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 128)
    trial.suggest_int("NUM_OUTPUT_LAYERS", 1, 3)
    trial.suggest_int("HIDDEN_SIZE", 16, 128)

    executor = optuna.integration.AllenNLPExecutor(trial, "undefined.jsonnet", "test")
    with pytest.raises(RuntimeError):
        executor.run()


def test_invalid_config_file() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 128)
    trial.suggest_int("NUM_OUTPUT_LAYERS", 1, 3)
    trial.suggest_int("HIDDEN_SIZE", 16, 128)

    executor = optuna.integration.AllenNLPExecutor(
        trial, "tests/integration_tests/allennlp_tests/invalid.jsonnet", "test"
    )
    with pytest.raises(RuntimeError):
        executor.run()


def test_invalid_param_name() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("_____DROPOUT", 0.0, 0.5)

    executor = optuna.integration.AllenNLPExecutor(
        trial, "tests/integration_tests/allennlp_tests/example.jsonnet", "test"
    )
    with pytest.raises(RuntimeError):
        executor.run()


def test_allennlp_executor() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("DROPOUT", 0.0, 0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial, "tests/integration_tests/allennlp_tests/example.jsonnet", tmp_dir
        )
        result = executor.run()
        assert isinstance(result, float)


def test_allennlp_executor_with_include_package() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("DROPOUT", 0.0, 0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial,
            "tests/integration_tests/allennlp_tests/example_with_include_package.jsonnet",
            tmp_dir,
            include_package="tests.integration_tests.allennlp_tests.tiny_single_id",
        )
        result = executor.run()
        assert isinstance(result, float)


def test_allennlp_executor_with_include_package_arr() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("DROPOUT", 0.0, 0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial,
            "tests/integration_tests/allennlp_tests/example_with_include_package.jsonnet",
            tmp_dir,
            include_package=["tests.integration_tests.allennlp_tests.tiny_single_id"],
        )
        result = executor.run()
        assert isinstance(result, float)


def test_allennlp_executor_with_options() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_float("DROPOUT", 0.0, 0.5)
    package_name = "tests.integration_tests.allennlp_tests.tiny_single_id"

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial,
            "tests/integration_tests/allennlp_tests/example_with_include_package.jsonnet",
            tmp_dir,
            force=True,
            file_friendly_logging=True,
            include_package=package_name,
        )

        # ``executor.run`` loads ``metrics.json``
        # after running ``allennlp.commands.train.train_model``.
        with open(os.path.join(executor._serialization_dir, "metrics.json"), "w") as fout:
            json.dump({executor._metrics: 1.0}, fout)

        expected_include_packages = [package_name, "optuna.integration.allennlp"]
        with mock.patch("allennlp.commands.train.train_model", return_value=None) as mock_obj:
            executor.run()
            assert mock_obj.call_args[1]["force"]
            assert mock_obj.call_args[1]["file_friendly_logging"]
            assert mock_obj.call_args[1]["include_package"] == expected_include_packages


def test_dump_best_config() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("DROPOUT", dropout, dropout)
            executor = optuna.integration.AllenNLPExecutor(trial, input_config_file, tmp_dir)
            return executor.run()

        dropout = 0.5
        input_config_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "example.jsonnet"
        )
        output_config_file = os.path.join(tmp_dir, "result.json")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        optuna.integration.allennlp.dump_best_config(input_config_file, output_config_file, study)
        best_config = json.loads(_jsonnet.evaluate_file(output_config_file))
        model_config = best_config["model"]
        target_config = model_config["text_field_embedder"]["token_embedders"]["token_characters"]
        assert target_config["dropout"] == dropout


def test_dump_best_config_with_environment_variables() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("DROPOUT", dropout, dropout)
            trial.suggest_float("LEARNING_RATE", 1e-2, 1e-1)
            executor = optuna.integration.AllenNLPExecutor(
                trial,
                input_config_file,
                tmp_dir,
                include_package="tests.integration_tests.allennlp_tests.tiny_single_id",
            )
            return executor.run()

        dropout = 0.5
        input_config_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "example_with_environment_variables.jsonnet",
        )
        output_config_file = os.path.join(tmp_dir, "result.json")

        os.environ["TRAIN_PATH"] = "tests/integration_tests/allennlp_tests/sentences.train"
        os.environ["VALID_PATH"] = "tests/integration_tests/allennlp_tests/sentences.valid"

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        optuna.integration.allennlp.dump_best_config(input_config_file, output_config_file, study)
        best_config = json.loads(_jsonnet.evaluate_file(output_config_file))
        assert os.getenv("TRAIN_PATH") == best_config["train_data_path"]
        assert os.getenv("VALID_PATH") == best_config["validation_data_path"]
        os.environ.pop("TRAIN_PATH")
        os.environ.pop("VALID_PATH")


def test_allennlp_pruning_callback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            reader = allennlp.data.dataset_readers.TextClassificationJsonReader(  # type: ignore[attr-defined]  # NOQA: E501
                tokenizer=allennlp.data.tokenizers.WhitespaceTokenizer(),  # type: ignore[attr-defined]  # NOQA: E501
            )
            data_loader = allennlp.data.data_loaders.MultiProcessDataLoader(
                reader=reader,
                data_path="tests/integration_tests/allennlp_tests/pruning_test.jsonl",
                batch_size=16,
            )
            vocab = allennlp.data.Vocabulary.from_instances(  # type: ignore[attr-defined]
                data_loader.iter_instances()
            )

            data_loader.index_with(vocab)

            embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder(  # type: ignore[attr-defined] # NOQA: E501
                {"tokens": allennlp.modules.Embedding(50, vocab=vocab)}  # type: ignore[attr-defined]  # NOQA: E501
            )
            encoder = allennlp.modules.seq2vec_encoders.GruSeq2VecEncoder(  # type: ignore[attr-defined]  # NOQA: E501
                input_size=50, hidden_size=50
            )
            model = allennlp.models.BasicClassifier(  # type: ignore[attr-defined]
                text_field_embedder=embedder, seq2vec_encoder=encoder, vocab=vocab
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            serialization_dir = os.path.join(tmp_dir, "trial_{}".format(trial.number))
            trainer = allennlp.training.GradientDescentTrainer(  # type: ignore[attr-defined]
                model=model,
                optimizer=optimizer,
                data_loader=data_loader,
                patience=None,
                num_epochs=1,
                serialization_dir=serialization_dir,
                callbacks=[AllenNLPPruningCallback(trial, "training_loss")],
            )
            trainer.train()
            return 1.0

        study = optuna.create_study(pruner=DeterministicPruner(True))
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED

        study = optuna.create_study(pruner=DeterministicPruner(False))
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value == 1.0


def test_allennlp_pruning_callback_with_invalid_storage() -> None:
    input_config_file = (
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner.jsonnet"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
            trial.suggest_float("DROPOUT", 0.0, 0.5)
            executor = optuna.integration.AllenNLPExecutor(trial, input_config_file, tmp_dir)
            return executor.run()

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            storage=None,
        )

        with pytest.raises(RuntimeError):
            study.optimize(objective)


@pytest.mark.parametrize(
    "pruner_class,pruner_kwargs",
    [
        (
            optuna.pruners.HyperbandPruner,
            {"min_resource": 3, "max_resource": 10, "reduction_factor": 5},
        ),
        (
            optuna.pruners.MedianPruner,
            {"n_startup_trials": 8, "n_warmup_steps": 1, "interval_steps": 3},
        ),
        (optuna.pruners.NopPruner, {}),
        (
            optuna.pruners.PercentilePruner,
            {"percentile": 50.0, "n_startup_trials": 10, "n_warmup_steps": 1, "interval_steps": 3},
        ),
        (
            optuna.pruners.SuccessiveHalvingPruner,
            {"min_resource": 3, "reduction_factor": 5, "min_early_stopping_rate": 1},
        ),
        (
            optuna.pruners.ThresholdPruner,
            {"lower": 0.0, "upper": 1.0, "n_warmup_steps": 3, "interval_steps": 2},
        ),
    ],
)
@pytest.mark.parametrize(
    "input_config_file",
    [
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner.jsonnet",
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner_distributed.jsonnet",  # noqa: E501
    ],
)
def test_allennlp_pruning_callback_with_executor(
    pruner_class: Type[optuna.pruners.BasePruner],
    pruner_kwargs: Dict[str, Union[int, float]],
    input_config_file: str,
) -> None:
    def run_allennlp_executor(pruner: optuna.pruners.BasePruner) -> None:
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage)
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.suggest_float("DROPOUT", 0.0, 0.5)
        executor = optuna.integration.AllenNLPExecutor(trial, input_config_file, serialization_dir)
        executor.run()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pruner_name = pruner_class.__name__
        os.mkdir(os.path.join(tmp_dir, pruner_name))
        storage = "sqlite:///" + os.path.join(tmp_dir, pruner_name, "result.db")
        serialization_dir = os.path.join(tmp_dir, pruner_name, "allennlp")

        pruner = pruner_class(**pruner_kwargs)
        run_allennlp_executor(pruner)
        process = psutil.Process()
        manager = _VariableManager(process.ppid())
        ret_pruner = _create_pruner(
            manager.get_value("pruner_class"),
            manager.get_value("pruner_kwargs"),
        )

        assert isinstance(ret_pruner, pruner_class)
        for key, value in pruner_kwargs.items():
            assert getattr(ret_pruner, "_{}".format(key)) == value


def test_allennlp_pruning_callback_with_invalid_executor() -> None:
    class SomeNewPruner(optuna.pruners.BasePruner):
        def __init__(self) -> None:
            pass

        def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
            return False

    input_config_file = (
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner.jsonnet"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = "sqlite:///" + os.path.join(tmp_dir, "result.db")
        serialization_dir = os.path.join(tmp_dir, "allennlp")
        pruner = SomeNewPruner()

        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage)
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.suggest_float("DROPOUT", 0.0, 0.5)

        with pytest.raises(ValueError):
            optuna.integration.AllenNLPExecutor(trial, input_config_file, serialization_dir)
