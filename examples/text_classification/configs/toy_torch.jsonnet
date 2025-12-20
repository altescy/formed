local ref(name) = { type: 'ref', ref: name };
local evaluator = {
  type: 'textclf.evaluators:ClassificationEvaluator',
  metrics: [
    { type: 'accuracy' },
    { type: 'fbeta' },
  ],
};

{
  steps: {
    train_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 1000,
      random_seed: 1,
    },
    val_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 50,
      random_seed: 2,
    },
    test_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 50,
      random_seed: 3,
    },
    datamodule: {
      type: 'ml::train_datamodule',
      datamodule: {
        type: 'textclf::text_classification',
        id: {},
        text: {
          type: 'tokenizer',
          surfaces: {},
        },
        label: {},
      },
      dataset: ref('train_dataset'),
    },
    model: {
      type: 'torch::train',
      model: {
        type: 'textclf::torch_text_classifier',
        num_classes: ref('datamodule.label.num_labels'),
        embedder: {
          type: 'analyzed_text',
          surface: {
            type: 'token',
            initializer: {
              type: 'xavier_uniform',
              shape: [
                ref('datamodule.text.surfaces.vocab_size'),  // vocab size
                32,  // embedding dim
              ],
            },
            padding_idx: ref('datamodule.text.surfaces.pad_index'),
          },
        },
        encoder: {
          type: 'lstm',
          input_dim: 32,
          hidden_dim: 32,
          bidirectional: false,
        },
        vectorizer: { type: 'boe', pooling: 'last' },
        dropout: 0.1,
      },
      trainer: {
        train_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'basic', batch_size: 32, drop_last: true },
          collator: ref('datamodule.batch'),
        },
        val_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'basic', batch_size: 32, drop_last: false },
          collator: ref('datamodule.batch'),
        },
        engine: {
          type: 'default',
          optimizer: { type: 'torch.optim:Adam', lr: 1e-3 },
        },
        callbacks: [
          { type: 'mlflow' },
          { type: 'evaluation', evaluator: evaluator },
          { type: 'early_stopping', patience: 3, metric: '+val/fbeta' },
        ],
        max_epochs: 10,
        logging_strategy: 'step',
        logging_interval: 5,
      },
      train_dataset: ref('train_dataset'),
      val_dataset: ref('test_dataset'),
    },
    test_metrics: {
      type: 'torch::evaluate',
      model: ref('model'),
      evaluator: evaluator,
      dataset: ref('test_dataset'),
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: {
          type: 'basic',
          batch_size: 32,
          shuffle: false,
          drop_last: false,
        },
        collator: ref('datamodule.batch'),
      },
      random_seed: 0,
    },
  },
}
