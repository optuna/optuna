import uuid

import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import optuna
import torch


DEVICE = -1  # If you want to use GPU, use DEVICE = 0

# Run tuning with small portion of data
# to reduce computational time.
# https://github.com/optuna/optuna/pull/949#pullrequestreview-364110499
MAX_DATA_SIZE = 800

GLOVE_FILE_PATH = (
    'https://s3-us-west-2.amazonaws.com/'
    'allennlp/datasets/glove/glove.6B.50d.txt.gz'
)


def prepare_data():
    glove_indexer = allennlp.data.token_indexers.SingleIdTokenIndexer(
        lowercase_tokens=True
    )
    tokenizer = allennlp.data.tokenizers.WordTokenizer(
        word_splitter=allennlp.data.tokenizers.word_splitter.JustSpacesWordSplitter(),
    )

    reader = allennlp.data.dataset_readers.TextClassificationJsonReader(
        token_indexers={'tokens': glove_indexer},
        tokenizer=tokenizer,
    )
    train_dataset = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl'
    )
    train_dataset = train_dataset[:MAX_DATA_SIZE]

    valid_dataset = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl'
    )
    valid_dataset = valid_dataset[:MAX_DATA_SIZE]

    vocab = allennlp.data.Vocabulary.from_instances(train_dataset)
    return train_dataset, valid_dataset, vocab


def create_model(vocab, trial: optuna.Trial):
    embedding = allennlp.modules.Embedding(
        embedding_dim=50,
        trainable=True,
        pretrained_file=GLOVE_FILE_PATH,
        num_embeddings=vocab.get_vocab_size('tokens'),
    )

    embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder(
        {'tokens': embedding}
    )

    output_dim = trial.suggest_int('output_dim', 80, 120)
    max_filter_size = trial.suggest_int('max_filter_size', 3, 6)
    num_filters = trial.suggest_int('num_filters', 64, 512)
    encoder = allennlp.modules.seq2vec_encoders.CnnEncoder(
        ngram_filter_sizes=range(1, max_filter_size),
        num_filters=num_filters,
        embedding_dim=50,
        output_dim=output_dim,
    )

    dropout = trial.suggest_uniform('dropout', 0, 0.5)
    model = allennlp.models.BasicClassifier(
        text_field_embedder=embedder,
        seq2vec_encoder=encoder,
        dropout=dropout,
        vocab=vocab,
    )

    return model


def objective(trial: optuna.Trial):
    train_dataset, valid_dataset, vocab = prepare_data()
    model = create_model(vocab, trial)

    if DEVICE > -1:
        print(f'send model to GPU #{DEVICE}')
        model.cuda(DEVICE)

    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    iterator = allennlp.data.iterators.BasicIterator(
        batch_size=32,
    )
    iterator.index_with(vocab)

    trainer = allennlp.training.Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        patience=1,
        num_epochs=5,
        cuda_device=DEVICE,
        serialization_dir=f'/tmp/xx/{uuid.uuid1()}',
    )
    metrics = trainer.train()
    return metrics['best_validation_accuracy']


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
