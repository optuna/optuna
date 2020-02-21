import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import optuna
import torch
import uuid


DEVICE = 0
GLOBE_FILE_PATH = (
    'https://s3-us-west-2.amazonaws.com/'
    'allennlp/datasets/glove/glove.6B.50d.txt.gz'
)


def prepare_data():
    globe_indexer = allennlp.data.token_indexers.SingleIdTokenIndexer(
        lowercase_tokens=True
    )
    tokenizer = allennlp.data.tokenizers.WordTokenizer(
        word_splitter=allennlp.data.tokenizers.word_splitter.JustSpacesWordSplitter(),
    )

    reader = allennlp.data.dataset_readers.TextClassificationJsonReader(
        token_indexers={'tokens': globe_indexer},
        tokenizer=tokenizer,
    )
    train_dataset = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl'
    )
    valid_dataset = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl'
    )
    test_dataset = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/test.jsonl'
    )

    vocab = allennlp.data.Vocabulary.from_instances(train_dataset)
    return train_dataset, valid_dataset, test_dataset, vocab


def objective(trial: optuna.Trial):
    train_dataset, valid_dataset, test_dataset, vocab = prepare_data()
    embedding = allennlp.modules.Embedding(
        embedding_dim=50,
        trainable=True,
        pretrained_file=GLOBE_FILE_PATH,
        num_embeddings=vocab.get_vocab_size('tokens'),
    )

    embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder(
        {'tokens': embedding}
    )

    output_dim = trial.suggest_int('output_dim', 10, 100)
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
        patience=3,
        num_epochs=10,
        cuda_device=DEVICE,
        serialization_dir=f'/tmp/xx/{uuid.uuid1()}',
    )
    metrics = trainer.train()
    return metrics['best_validation_accuracy']


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
