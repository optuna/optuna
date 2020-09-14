import json
import os
import tempfile
from typing import Dict
from typing import Union

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
from torch.utils.data import DataLoader

import optuna
from optuna.integration.allennlp import AllenNLPPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_build_params() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
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
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
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
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-2)
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
    assert params["trainer"]["lr"] == 1e-2
    path = params["model"]["text_field_embedder"]["token_embedders"]["token_characters"]["dropout"]
    assert path == 0.0


def test_missing_config_file() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
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
    trial.suggest_uniform("LEARNING_RATE", 1e-2, 1e-1)
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)
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
    trial.suggest_uniform("_____DROPOUT", 0.0, 0.5)

    executor = optuna.integration.AllenNLPExecutor(
        trial, "tests/integration_tests/allennlp_tests/example.jsonnet", "test"
    )
    with pytest.raises(RuntimeError):
        executor.run()


def test_allennlp_executor() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial, "tests/integration_tests/allennlp_tests/example.jsonnet", tmp_dir
        )
        result = executor.run()
        assert isinstance(result, float)


def test_allennlp_executor_with_include_package() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)

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
    trial.suggest_uniform("DROPOUT", 0.0, 0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        executor = optuna.integration.AllenNLPExecutor(
            trial,
            "tests/integration_tests/allennlp_tests/example_with_include_package.jsonnet",
            tmp_dir,
            include_package=["tests.integration_tests.allennlp_tests.tiny_single_id"],
        )
        result = executor.run()
        assert isinstance(result, float)


def test_dump_best_config() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_uniform("DROPOUT", dropout, dropout)
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


def test_allennlp_pruning_callback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        def objective(trial: optuna.Trial) -> float:
            reader = allennlp.data.dataset_readers.TextClassificationJsonReader(
                tokenizer=allennlp.data.tokenizers.SpacyTokenizer()
            )
            dataset = reader.read("tests/integration_tests/allennlp_tests/pruning_test.jsonl")
            vocab = allennlp.data.Vocabulary.from_instances(dataset)
            dataset.index_with(vocab)
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
            data_loader = DataLoader(
                dataset, batch_size=10, collate_fn=allennlp.data.allennlp_collate
            )
            serialization_dir = os.path.join(tmp_dir, "trial_{}".format(trial.number))
            trainer = allennlp.training.GradientDescentTrainer(
                model=model,
                optimizer=optimizer,
                data_loader=data_loader,
                patience=None,
                num_epochs=1,
                serialization_dir=serialization_dir,
                epoch_callbacks=[AllenNLPPruningCallback(trial, "training_loss")],
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


@pytest.mark.parametrize(
    "pruner_name,pruner_kwargs",
    [
        ("hyperband", {"min_resource": 3, "max_resource": 10, "reduction_factor": 5}),
        ("median", {"n_startup_trials": 8, "n_warmup_steps": 1, "interval_steps": 3}),
        ("noop", {}),
        ("percentile", {"percentile": 50.0, "n_startup_trials": 10, "n_warmup_steps": 1, "interval_steps": 3}),  # NOQA
        ("successive_halving", {"min_resource": 3, "reduction_factor": 5, "min_early_stopping_rate": 1}),  # NOQA
        ("threshold", {"lower": 0.0, "upper": 1.0, "n_warmup_steps": 3, "interval_steps": 2})
    ]
)
def test_allennlp_pruning_callback_with_executor(
        pruner_name: str,
        pruner_kwargs: Dict[str, Union[int, float]]
) -> None:
    input_config_file = (
        "tests/integration_tests/allennlp_tests/example_with_executor_and_pruner.jsonnet"
    )

    def run_allennlp_executor(pruner: optuna.pruners.BasePruner) -> None:
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            storage=storage,
        )
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.suggest_float("DROPOUT", 0.0, 0.5)
        executor = optuna.integration.AllenNLPExecutor(
            trial,
            input_config_file,
            serialization_dir
        )
        executor.run()

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.mkdir(os.path.join(tmp_dir, pruner_name))
        storage = "sqlite:///" + os.path.join(tmp_dir, pruner_name, "result.db")
        serialization_dir = os.path.join(tmp_dir, pruner_name, "allennlp")

        if pruner_name == "hyperband":
            hyperband_pruner = optuna.pruners.HyperbandPruner(
                min_resource=int(pruner_kwargs["min_resource"]),
                max_resource=int(pruner_kwargs["max_resource"]),
                reduction_factor=int(pruner_kwargs["reduction_factor"]),
            )
            run_allennlp_executor(hyperband_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.HyperbandPruner)
            assert pruner._min_resource == pruner_kwargs["min_resource"]
            assert pruner._max_resource == pruner_kwargs["max_resource"]
            assert pruner._reduction_factor == pruner_kwargs["reduction_factor"]

        elif pruner_name == "median":
            median_pruner = optuna.pruners.MedianPruner(
                n_startup_trials=int(pruner_kwargs["n_startup_trials"]),
                n_warmup_steps=int(pruner_kwargs["n_warmup_steps"]),
                interval_steps=int(pruner_kwargs["interval_steps"]),
            )
            run_allennlp_executor(median_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.MedianPruner)
            assert pruner._n_startup_trials == pruner_kwargs["n_startup_trials"]
            assert pruner._n_warmup_steps == pruner_kwargs["n_warmup_steps"]
            assert pruner._interval_steps == pruner_kwargs["interval_steps"]

        elif pruner_name == "noop":
            noop_pruner = optuna.pruners.NopPruner()
            run_allennlp_executor(noop_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.NopPruner)

        elif pruner_name == "percentile":
            percentile_pruner = optuna.pruners.PercentilePruner(
                percentile=float(pruner_kwargs["percentile"]),
                n_startup_trials=int(pruner_kwargs["n_startup_trials"]),
                n_warmup_steps=int(pruner_kwargs["n_warmup_steps"]),
                interval_steps=int(pruner_kwargs["interval_steps"]),
            )
            run_allennlp_executor(percentile_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.PercentilePruner)
            assert pruner._percentile == pruner_kwargs["percentile"]
            assert pruner._n_startup_trials == pruner_kwargs["n_startup_trials"]
            assert pruner._n_warmup_steps == pruner_kwargs["n_warmup_steps"]
            assert pruner._interval_steps == pruner_kwargs["interval_steps"]

        elif pruner_name == "successive_halving":
            successive_halving_pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=int(pruner_kwargs["min_resource"]),
                reduction_factor=int(pruner_kwargs["reduction_factor"]),
                min_early_stopping_rate=int(pruner_kwargs["min_early_stopping_rate"]),
            )
            run_allennlp_executor(successive_halving_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)
            assert pruner._min_resource == pruner_kwargs["min_resource"]
            assert pruner._reduction_factor == pruner_kwargs["reduction_factor"]
            assert pruner._min_early_stopping_rate == pruner_kwargs["min_early_stopping_rate"]

        elif pruner_name == "threshold":
            threshold_pruner = optuna.pruners.ThresholdPruner(
                lower=float(pruner_kwargs["lower"]),
                upper=float(pruner_kwargs["upper"]),
                n_warmup_steps=int(pruner_kwargs["n_warmup_steps"]),
                interval_steps=int(pruner_kwargs["interval_steps"]),
            )
            run_allennlp_executor(threshold_pruner)
            pruner = optuna.integration.allennlp._create_pruner()

            assert isinstance(pruner, optuna.pruners.ThresholdPruner)
            assert pruner._lower == pruner_kwargs["lower"]
            assert pruner._upper == pruner_kwargs["upper"]
            assert pruner._n_warmup_steps == pruner_kwargs["n_warmup_steps"]
            assert pruner._interval_steps == pruner_kwargs["interval_steps"]


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

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            storage=storage,
        )
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.suggest_float("DROPOUT", 0.0, 0.5)

        with pytest.raises(ValueError):
            optuna.integration.AllenNLPExecutor(
                trial,
                input_config_file,
                serialization_dir
            )


def test_infer_and_cast() -> None:
    assert optuna.integration.allennlp._infer_and_cast(None) is None
    assert optuna.integration.allennlp._infer_and_cast("True") is True
    assert optuna.integration.allennlp._infer_and_cast("False") is False
    assert optuna.integration.allennlp._infer_and_cast("3.14") == 3.14
    assert optuna.integration.allennlp._infer_and_cast("42") == 42
    assert optuna.integration.allennlp._infer_and_cast("auto") == "auto"
