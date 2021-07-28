import json
import os
import tempfile
from unittest import mock

import _jsonnet
import allennlp.data
import allennlp.data.dataset_readers
import allennlp.data.tokenizers
import allennlp.models
import allennlp.modules
import allennlp.modules.seq2vec_encoders
import allennlp.modules.text_field_embedders
import allennlp.training
import pytest
import torch.optim

import optuna
from optuna.integration.allennlp import AllenNLPPruningCallback
from optuna.testing.integration import DeterministicPruner


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

    # Optuna trial overwrites a parameter specified by environment variable
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
        # after running ``optuna.integration.allennlp._train.train_model_with_optuna``.
        with open(os.path.join(executor._serialization_dir, "metrics.json"), "w") as fout:
            json.dump({executor._metrics: 1.0}, fout)

        patch_target = "optuna.integration.allennlp._train.train_model_with_optuna"
        with mock.patch(patch_target, return_value=None) as mock_obj:
            executor.run()
            assert mock_obj.call_args[1]["force"]
            assert mock_obj.call_args[1]["file_friendly_logging"]
            assert mock_obj.call_args[1]["include_package"] == [package_name]


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
            reader = allennlp.data.dataset_readers.TextClassificationJsonReader(
                tokenizer=allennlp.data.tokenizers.WhitespaceTokenizer(),
            )
            data_loader = allennlp.data.data_loaders.MultiProcessDataLoader(
                reader=reader,
                data_path="tests/integration_tests/allennlp_tests/pruning_test.jsonl",
                batch_size=16,
            )
            vocab = allennlp.data.Vocabulary.from_instances(data_loader.iter_instances())
            data_loader.index_with(vocab)

            embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder(
                {"tokens": allennlp.modules.Embedding(50, vocab=vocab)}
            )
            encoder = allennlp.modules.seq2vec_encoders.GruSeq2VecEncoder(
                input_size=50, hidden_size=50
            )
            model = allennlp.models.BasicClassifier(
                text_field_embedder=embedder, seq2vec_encoder=encoder, vocab=vocab
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            serialization_dir = os.path.join(tmp_dir, "trial_{}".format(trial.number))
            trainer = allennlp.training.GradientDescentTrainer(
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


def test_allennlp_pruning_callback_monitor() -> None:
    study = optuna.study.create_study()
    trial = study.ask()

    pruning_callback = AllenNLPPruningCallback(trial, "best_validation_loss")
    assert pruning_callback._monitor == "best_validation_loss"

    trial.set_system_attr("allennlp:monitor", "best_validation_accuracy")
    pruning_callback_without_monitor = AllenNLPPruningCallback(trial)
    pruning_callback_without_monitor._monitor == "best_validation_accuracy"


def test_allennlp_pruning_callback_with_executor() -> None:
    input_config_file = (
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner.jsonnet"
    )

    def run_allennlp_executor(pruner: optuna.pruners.BasePruner) -> None:
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage)
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.suggest_float("DROPOUT", 0.0, 0.5)
        executor = optuna.integration.AllenNLPExecutor(trial, input_config_file, serialization_dir)
        executor.run()

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = "sqlite:///" + os.path.join(tmp_dir, "result.db")
        serialization_dir = os.path.join(tmp_dir, "allennlp")

        pruner_mock = mock.Mock()
        pruner_mock.prune = mock.Mock(return_value=False)
        run_allennlp_executor(pruner_mock)
        pruner_mock.prune.assert_called_once()
