"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of sentiment classification using AllenNLP.
Since it is too time-consuming to use the entire dataset, we here use a small subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python allennlp_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize allennlp_simple.py objective --n-trials=100 --study-name $STUDY_NAME \
      --storage sqlite:///example.db

"""

import os
import random
import shutil

import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import numpy
import torch

import optuna
from optuna.integration import AllenNLPPruningCallback


DEVICE = -1  # If you want to use GPU, use DEVICE = 0.
MAX_DATA_SIZE = 3000
MODEL_DIR = os.path.join(os.getcwd(), "result")
TARGET_METRIC = "accuracy"


class SubsampledDatasetReader(allennlp.data.dataset_readers.TextClassificationJsonReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read(self, datapath):
        data = list(super()._read(datapath))
        random.shuffle(data)
        yield from data[:MAX_DATA_SIZE]


def prepare_data():
    glove_indexer = allennlp.data.token_indexers.SingleIdTokenIndexer(lowercase_tokens=True)
    tokenizer = allennlp.data.tokenizers.whitespace_tokenizer.WhitespaceTokenizer()

    reader = SubsampledDatasetReader(
        token_indexers={"tokens": glove_indexer}, tokenizer=tokenizer,
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
    output_dim = trial.suggest_int("output_dim", 16, 128)
    max_filter_size = trial.suggest_int("max_filter_size", 3, 6)
    num_filters = trial.suggest_int("num_filters", 16, 128)
    encoder = allennlp.modules.seq2vec_encoders.CnnEncoder(
        ngram_filter_sizes=range(1, max_filter_size),
        num_filters=num_filters,
        embedding_dim=50,
        output_dim=output_dim,
    )

    embedding = allennlp.modules.Embedding(
        embedding_dim=50,
        trainable=True,
        pretrained_file="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",  # NOQA
        vocab=vocab,
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

    lr = trial.suggest_float("lr", 1e-1, 1e0, log=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, collate_fn=allennlp.data.allennlp_collate
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
        num_epochs=50,
        cuda_device=DEVICE,
        serialization_dir=serialization_dir,
        epoch_callbacks=[AllenNLPPruningCallback(trial, "validation_" + TARGET_METRIC)],
    )
    metrics = trainer.train()
    return metrics["best_validation_" + TARGET_METRIC]


if __name__ == "__main__":
    random.seed(41)
    torch.manual_seed(41)
    numpy.random.seed(41)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
