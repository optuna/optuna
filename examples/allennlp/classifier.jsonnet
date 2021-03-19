// Use dev.jsonl for training to reduce computation time.
local TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';
local VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/test.jsonl';

// std.parseJson for floating point.
local DROPOUT = std.parseJson(std.extVar('DROPOUT'));
local EMBEDDING_DIM = std.parseInt(std.extVar('EMBEDDING_DIM'));
local CNN_FIELDS(max_filter_size, embedding_dim, hidden_size, num_filters) = {
  type: 'cnn',
  ngram_filter_sizes: std.range(1, max_filter_size),
  num_filters: num_filters,
  embedding_dim: embedding_dim,
  output_dim: hidden_size,
};

// You have to use parseInt for MAX_FILTER_SIZE
// since it is used as an argument of the built-in function std.range in CNN_FIELDS.
local ENCODER = CNN_FIELDS(
  std.parseInt(std.extVar('MAX_FILTER_SIZE')),
  EMBEDDING_DIM,
  std.parseInt(std.extVar('HIDDEN_SIZE')),
  std.parseInt(std.extVar('NUM_FILTERS'))
);


{
  numpy_seed: 42,
  pytorch_seed: 42,
  random_seed: 42,
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: TRAIN_PATH,
  validation_data_path: VALIDATION_PATH,
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: EMBEDDING_DIM,
        },
      },
    },
    seq2vec_encoder: ENCODER,
    dropout: DROPOUT,
  },
  data_loader: {
    batch_size: 64,
  },

  trainer: {
    cuda_device: -1,
    num_epochs: 5,
    optimizer: {
      lr: 0.1,
      type: 'adam',
    },
    patience: 2,
    validation_metric: '+accuracy',
    callbacks: [
      {
        type: 'optuna_pruner',
      },
    ],
  },
}
