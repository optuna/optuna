local DROPOUT = std.parseJson(std.extVar('DROPOUT'));
local LEARNING_RATE = std.parseJson(std.extVar('LEARNING_RATE'));


{
  dataset_reader: {
    type: 'sequence_tagging',
    word_tag_delimiter: '/',
    token_indexers: {
      tokens: {
        type: 'tiny_single_id',
        lowercase_tokens: true,
      },
      token_characters: {
        type: 'characters',
      },
    },
  },
  train_data_path: std.extVar('TRAIN_PATH'),
  validation_data_path: std.extVar('VALID_PATH'),
  model: {
    type: 'simple_tagger',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 5,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            embedding_dim: 4,
          },
          encoder: {
            type: 'cnn',
            embedding_dim: 4,
            num_filters: 5,
            ngram_filter_sizes: [3],
          },
          dropout: DROPOUT,
        },
      },
    },
    encoder: {
      type: 'lstm',
      input_size: 10,
      hidden_size: 10,
      num_layers: 2,
      dropout: 0,
      bidirectional: true,
    },
  },
  data_loader: {
    batch_size: 32,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: LEARNING_RATE,
    },
    num_epochs: 1,
    patience: 10,
    cuda_device: -1,
  },
}
