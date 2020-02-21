import allennlp
import allennlp.data
import allennlp.models
import allennlp.modules
import optuna
import random
import torch
import uuid


globe_indexer = allennlp.data.token_indexers.SingleIdTokenIndexer(lowercase_tokens=True)

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
vocab_size = vocab.get_vocab_size('tokens')

globe_file_path = (
    'https://s3-us-west-2.amazonaws.com/'
    'allennlp/datasets/glove/glove.6B.50d.txt.gz'
)

device = -1


def objective(trial: optuna.Trial):
    embedding = allennlp.modules.Embedding(
        embedding_dim=50,
        trainable=True,
        pretrained_file=globe_file_path,
        num_embeddings=vocab.get_vocab_size('tokens'),
    )

    embedder = allennlp.modules.text_field_embedders.BasicTextFieldEmbedder(
        {'tokens': embedding}
    )

    encoder_output_dim = trial.suggest_int('output_dim', 10, 100)
    encoder = allennlp.modules.seq2vec_encoders.CnnEncoder(
        ngram_filter_sizes=range(1, 3),
        num_filters=3,
        embedding_dim=50,
        output_dim=encoder_output_dim,
    )

    model = allennlp.models.BasicClassifier(
        text_field_embedder=embedder,
        seq2vec_encoder=encoder,
        vocab=vocab
    )

    if device > -1:
        print(f'send model to GPU #{device}')
        model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    iterator = allennlp.data.iterators.BasicIterator(
        batch_size=10,
    )
    iterator.index_with(vocab)

    trainer = allennlp.training.Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        patience=10,
        num_epochs=1000,
        cuda_device=device,
        serialization_dir=f'/tmp/xx/{uuid.uuid1()}',
    )
    trainer.train()
    return random.random()


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)

    print(study.best_params)
