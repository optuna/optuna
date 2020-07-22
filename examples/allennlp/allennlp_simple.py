"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of AllenTune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of sentiment classification using AllenNLP.
Since it is too time-consuming to use the entire dataset, we here use a small subset of it.

"""

import os
import pkg_resources
import random
import shutil
import sys

import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import numpy
import torch

import optuna
from optuna.integration import AllenNLPPruningCallback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from subsample_dataset_reader import SubsampleDatasetReader  # NOQA


DEVICE = -1  # If you want to use GPU, use DEVICE = 0.
N_TRAIN_DATA_SIZE = 2000
N_VALIDATION_DATA_SIZE = 1000
MODEL_DIR = os.path.join(os.getcwd(), "result")
TARGET_METRIC = "accuracy"


def prepare_data():
    indexer = allennlp.data.token_indexers.SingleIdTokenIndexer(lowercase_tokens=True)
    tokenizer = allennlp.data.tokenizers.whitespace_tokenizer.WhitespaceTokenizer()

    reader = SubsampleDatasetReader(
        token_indexers={"tokens": indexer},
        tokenizer=tokenizer,
        train_data_size=N_TRAIN_DATA_SIZE,
        validation_data_size=N_VALIDATION_DATA_SIZE,
    )
    train_dataset = reader.read(
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl"
    )
    valid_dataset = reader.read(
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl"
    )
    vocab = allennlp.data.Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    return train_dataset, valid_dataset, vocab


def create_model(vocab, trial):
    dropout = trial.suggest_float("dropout", 0, 0.5)
    embedding_dim = trial.suggest_int("embedding_dim", 16, 128)
    output_dim = trial.suggest_int("output_dim", 32, 128)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 4)
    num_filters = trial.suggest_int("num_filters", 32, 128)

    embedding = allennlp.modules.Embedding(
        embedding_dim=embedding_dim, trainable=True, vocab=vocab,
    )

    encoder = allennlp.modules.seq2vec_encoders.CnnEncoder(
        ngram_filter_sizes=range(2, max_filter_size),
        num_filters=num_filters,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
    )

    embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder({"tokens": embedding})
    model = allennlp.models.BasicClassifier(
        text_field_embedder=embedder, seq2vec_encoder=encoder, dropout=dropout, vocab=vocab,
    )

    return model


def objective(trial):
    train_dataset, valid_dataset, vocab = prepare_data()
    model = create_model(vocab, trial)

    if DEVICE > -1:
        model.to(torch.device("cuda:{}".format(DEVICE)))

    lr = trial.suggest_float("lr", 1e-2, 1e-1, log=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, collate_fn=allennlp.data.allennlp_collate
    )
    validation_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=64, collate_fn=allennlp.data.allennlp_collate
    )

    serialization_dir = os.path.join(MODEL_DIR, "trial_{}".format(trial.number))
    trainer = allennlp.training.GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        validation_data_loader=validation_data_loader,
        validation_metric="+" + TARGET_METRIC,
        patience=None,  # `patience=None` since it could conflict with AllenNLPPruningCallback
        num_epochs=30,
        cuda_device=DEVICE,
        serialization_dir=serialization_dir,
        epoch_callbacks=[AllenNLPPruningCallback(trial, "validation_" + TARGET_METRIC)],
    )
    metrics = trainer.train()
    return metrics["best_validation_" + TARGET_METRIC]


if __name__ == "__main__":
    if pkg_resources.parse_version(allennlp.__version__) < pkg_resources.parse_version("1.0.0"):
        raise RuntimeError("AllenNLP>=1.0.0 is required for this example.")

    random.seed(41)
    torch.manual_seed(41)
    numpy.random.seed(41)

    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
